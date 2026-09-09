import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import sys
import tarfile
import time
from pathlib import Path

import xxhash
import zstandard as zstd

MAGIC = b'\x28\xb5\x2f\xfd'
CHUNK = 8 * 1024 * 1024


def save_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


class TarScanner:
    """Incrementally validate ordinary PAX tar headers and hash file contents."""
    def __init__(self):
        self.header = bytearray()
        self.info = None
        self.remaining = 0
        self.content_remaining = 0
        self.metadata = bytearray()
        self.pax = {}
        self.global_pax = {}
        self.entries = []
        self.position = 0
        self.digest = None
        self.ended = False

    def finish(self, callback):
        info = self.info
        if info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE):
            values = {}
            pos = 0
            data = bytes(self.metadata)
            while pos < len(data):
                space = data.index(b' ', pos)
                length = int(data[pos:space])
                assert length > space - pos + 1 and pos + length <= len(data)
                key, value = data[space + 1:pos + length - 1].split(b'=', 1)
                values[key.decode('utf-8')] = value.decode('utf-8', 'surrogateescape')
                pos += length
            if info.type == tarfile.XGLTYPE:
                self.global_pax.update(values)
            else:
                self.pax.update(values)
        elif info.type in (tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
            key = 'path' if info.type == tarfile.GNUTYPE_LONGNAME else 'linkpath'
            self.pax[key] = bytes(self.metadata).rstrip(b'\x00').decode('utf-8', 'surrogateescape')
        else:
            record = {'name': info.name.rstrip('/'), 'size': info.size,
                      'mtime': info.mtime, 'is_directory': info.isdir(),
                      'sha256': self.digest.hexdigest() if self.digest else None}
            assert record['name'].startswith('speechtokenizer/')
            relative = record['name'].split('/', 1)[1]
            assert relative.split('/')[0] not in ('data', '.git')
            assert relative != 'train-clean-100.tar.gz'
            self.entries.append(record)
            self.pax.clear()
            callback(self.position)
        self.info = None
        self.metadata.clear()
        self.digest = None

    def feed(self, data, callback):
        pos = 0
        view = memoryview(data)
        while pos < len(data):
            if self.ended:
                assert not any(view[pos:]), 'Nonzero bytes after tar end'
                self.position += len(data) - pos
                return
            if self.info is None:
                count = min(512 - len(self.header), len(data) - pos)
                self.header.extend(view[pos:pos + count])
                pos += count
                self.position += count
                if len(self.header) < 512:
                    continue
                header = bytes(self.header)
                self.header.clear()
                if header == b'\x00' * 512:
                    self.ended = True
                    continue
                self.info = tarfile.TarInfo.frombuf(header, 'utf-8', 'surrogateescape')
                meta = self.info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE,
                                          tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK)
                if not meta:
                    values = dict(self.global_pax, **self.pax)
                    if 'path' in values:
                        self.info.name = values['path']
                    if 'size' in values:
                        self.info.size = int(values['size'])
                    if 'mtime' in values:
                        self.info.mtime = float(values['mtime'])
                    self.digest = hashlib.sha256() if self.info.isfile() else None
                self.content_remaining = self.info.size
                self.remaining = (self.info.size + 511) // 512 * 512
                if not self.remaining:
                    self.finish(callback)
            else:
                count = min(self.remaining, len(data) - pos)
                content_count = min(count, self.content_remaining)
                if self.digest is not None:
                    self.digest.update(view[pos:pos + content_count])
                elif self.info.type in (tarfile.XHDTYPE, tarfile.XGLTYPE,
                                        tarfile.GNUTYPE_LONGNAME, tarfile.GNUTYPE_LONGLINK):
                    self.metadata.extend(view[pos:pos + content_count])
                self.content_remaining -= content_count
                self.remaining -= count
                self.position += count
                pos += count
                if not self.remaining:
                    self.finish(callback)


def inspect_partial(path, report):
    size = path.stat().st_size
    scanner = TarScanner()
    checkpoint = None
    frames = 0
    last_report = 0
    with path.open('rb', buffering=CHUNK) as raw:
        while raw.tell() < size:
            frame_start = raw.tell()
            sample = raw.read(18)
            if len(sample) < 6:
                break
            assert sample[:4] == MAGIC, f'Invalid Zstandard frame at {frame_start}'
            header_size = zstd.frame_header_size(sample)
            params = zstd.get_frame_parameters(sample)
            assert params.content_size == zstd.CONTENTSIZE_UNKNOWN
            raw.seek(frame_start + header_size)
            decoder = zstd.ZstdDecompressor().decompressobj()
            decoder.decompress(sample[:header_size])
            frame_hash = xxhash.xxh64()
            frame_finished = False
            while raw.tell() < size:
                block_start = raw.tell()
                header = raw.read(3)
                if len(header) != 3:
                    break
                value = int.from_bytes(header, 'little')
                block_type = (value >> 1) & 3
                block_size = value >> 3
                assert block_type != 3 and block_size <= 131072
                physical_size = 1 if block_type == 1 else block_size
                body = raw.read(physical_size)
                if len(body) != physical_size:
                    break
                before_hash = frame_hash.intdigest()
                decoded = decoder.decompress(header + body)
                decoded_start = scanner.position

                def completed(end_position):
                    nonlocal checkpoint
                    checkpoint = {
                        'compressed_cut': block_start,
                        'frame_checksum': before_hash,
                        'has_checksum': params.has_checksum,
                        'fragment': decoded[:end_position - decoded_start],
                        'tar_end': end_position,
                        'entries': len(scanner.entries),
                    }

                scanner.feed(decoded, completed)
                frame_hash.update(decoded)
                if time.monotonic() - last_report > 15:
                    report(stage='recovering', percent=round(raw.tell() / size * 100, 2),
                           scanned_bytes=raw.tell(), scan_total_bytes=size,
                           recovered_source_bytes=sum(e['size'] for e in scanner.entries),
                           recovered_entries=len(scanner.entries))
                    last_report = time.monotonic()
                if value & 1:
                    checksum = raw.read(4) if params.has_checksum else b''
                    if params.has_checksum:
                        if len(checksum) < 4:
                            break
                        assert int.from_bytes(checksum, 'little') == frame_hash.intdigest() & 0xffffffff
                        decoder.decompress(checksum)
                    assert decoder.eof
                    frame_finished = True
                    frames += 1
                    break
            if not frame_finished:
                break
    assert checkpoint is not None, 'No complete tar entries recovered'
    assert checkpoint['entries'] == len(scanner.entries)
    checkpoint['original_size'] = size
    checkpoint['validated_frames'] = frames
    return checkpoint, scanner.entries


class HashReader:
    def __init__(self, stream, progress=None):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.count = 0
        self.progress = progress

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        self.count += len(data)
        if self.progress:
            self.progress(self.count)
        return data


def enumerate_current(root, recovered):
    seen = {entry['name']: entry for entry in recovered}
    pending = []
    changes = []
    for base, dirs, files in os.walk(root):
        if Path(base) == root:
            dirs[:] = [d for d in dirs if d not in ('data', '.git') and
                       not d.startswith('speechtokenizer_no_datasets_')]
        for name in sorted(dirs + files):
            path = Path(base) / name
            if name.startswith('~$') and path.suffix.lower() in (
                    '.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls', '.xlsm', '.pptm', '.docm'):
                continue
            if path.parent == root and (name in ('train-clean-100.tar.gz', '查看压缩进度.ps1') or
                                        name.startswith('speechtokenizer_no_datasets_')):
                continue
            archive_name = 'speechtokenizer/' + path.relative_to(root).as_posix()
            info = path.lstat()
            old = seen.get(archive_name)
            if old and (path.is_dir() or
                        (info.st_size == old['size'] and abs(info.st_mtime - old['mtime']) < 0.000001)):
                continue
            if old:
                changes.append(archive_name)
            pending.append((path, archive_name))
    return pending, changes


def resume(root, archive, report):
    partial = Path(str(archive) + '.partial')
    checkpoint, entries = inspect_partial(partial, report)
    pending, replacements = enumerate_current(root, entries)
    retained_source = sum(e['size'] for e in entries)
    pending_size = sum(p.stat().st_size for p, _ in pending if p.is_file())
    total = retained_source + pending_size
    backup = Path(str(archive) + '.resume-tail.backup')
    if backup.exists():
        backup = Path(str(archive) + f'.resume-tail-{time.time_ns()}.backup')
    recovery_info = Path(str(archive) + '.recovery.json')
    fragment_path = Path(str(archive) + '.recovery-fragment.bin')
    fragment_path.write_bytes(checkpoint['fragment'])
    save_json(Path(str(archive) + '.recovered-entries.json'), entries)
    with partial.open('rb') as source, backup.open('xb') as target:
        source.seek(checkpoint['compressed_cut'])
        shutil.copyfileobj(source, target, CHUNK)
        target.flush()
        os.fsync(target.fileno())
    save_json(recovery_info, {**{k: v for k, v in checkpoint.items() if k != 'fragment'},
                             'retained_source_bytes': retained_source,
                             'pending_source_bytes': pending_size,
                             'pending_entries': len(pending), 'replacements': replacements,
                             'tail_backup': str(backup)})
    report(stage='compressing', percent=round(retained_source / total * 100, 2),
           processed_source_bytes=retained_source, total_source_bytes=total,
           remaining_entries=len(pending))
    done = retained_source
    last_report = 0
    changed_during_pack = []
    with partial.open('r+b') as raw:
        raw.seek(checkpoint['compressed_cut'])
        raw.truncate()
        raw.write(b'\x01\x00\x00')  # Empty final raw block.
        if checkpoint['has_checksum']:
            raw.write(struct.pack('<I', checkpoint['frame_checksum'] & 0xffffffff))
        raw.flush()
        with zstd.ZstdCompressor(level=1, threads=4, write_checksum=True).stream_writer(raw, closefd=False) as compressed:
            compressed.write(checkpoint['fragment'])
            with tarfile.open(fileobj=compressed, mode='w|', bufsize=CHUNK) as tf:
                tf.copybufsize = CHUNK
                for path, archive_name in pending:
                    before = path.lstat()
                    info = tf.gettarinfo(str(path), arcname=archive_name)

                    def file_progress(count):
                        nonlocal last_report
                        if time.monotonic() - last_report > 15:
                            report(stage='compressing', percent=round((done + count) / total * 100, 2),
                                   processed_source_bytes=done + count, total_source_bytes=total,
                                   archive_bytes=raw.tell(), current_file=archive_name)
                            last_report = time.monotonic()

                    if info.isfile():
                        with path.open('rb') as source:
                            reader = HashReader(source, file_progress)
                            tf.addfile(info, reader)
                        digest = reader.digest.hexdigest()
                    else:
                        tf.addfile(info)
                        digest = None
                    entries.append({'name': info.name.rstrip('/'), 'size': info.size,
                                    'mtime': info.mtime, 'is_directory': info.isdir(), 'sha256': digest})
                    after = path.lstat()
                    if after.st_size != before.st_size or after.st_mtime_ns != before.st_mtime_ns:
                        changed_during_pack.append(archive_name)
                    done += info.size
        raw.flush()
        os.fsync(raw.fileno())
    manifest = {'archive': str(archive), 'entries': entries,
                'excluded': ['data/', 'train-clean-100.tar.gz', '.git/', 'Office temporary lock files (~$)',
                             'packaging helpers and generated archives'],
                'replacements': replacements, 'changed_during_pack': changed_during_pack,
                'recovery_backup': str(backup),
                'source_bytes': total, 'archive_bytes': partial.stat().st_size}
    save_json(Path(str(archive) + '.manifest.json'), manifest)
    report(stage='verifying', percent=0, archive_bytes=partial.stat().st_size)


def verify(archive, report):
    partial = Path(str(archive) + '.partial')
    manifest = json.loads(Path(str(archive) + '.manifest.json').read_text(encoding='utf-8'))
    expected = manifest['entries']
    total = partial.stat().st_size
    last_report = 0
    count = 0

    def progress(compressed_bytes):
        nonlocal last_report
        if time.monotonic() - last_report > 15:
            report(stage='verifying', percent=round(compressed_bytes / total * 100, 2),
                   verified_compressed_bytes=compressed_bytes, archive_bytes=total, verified_entries=count)
            last_report = time.monotonic()

    with partial.open('rb') as raw:
        hashed = HashReader(raw, progress)
        with zstd.ZstdDecompressor().stream_reader(hashed, read_across_frames=True) as stream:
            with tarfile.open(fileobj=stream, mode='r|', bufsize=CHUNK) as tf:
                for member in tf:
                    assert count < len(expected), 'Unexpected archive member'
                    entry = expected[count]
                    assert member.name.rstrip('/') == entry['name'], f'Path mismatch at {count}'
                    assert member.size == entry['size']
                    assert member.isdir() == entry['is_directory']
                    if member.isfile():
                        digest = hashlib.sha256()
                        with tf.extractfile(member) as content:
                            while True:
                                chunk = content.read(CHUNK)
                                if not chunk:
                                    break
                                digest.update(chunk)
                        assert digest.hexdigest() == entry['sha256'], f'Hash mismatch: {member.name}'
                    count += 1
            while stream.read(CHUNK):
                pass
        archive_hash = hashed.digest.hexdigest()
    assert count == len(expected), 'Missing archive members'
    partial.rename(archive)
    result = {'success': True, 'archive': str(archive), 'verified_entries': count,
              'sha256': archive_hash, 'compressed_stream_checksum': 'passed',
              'all_file_sha256': 'passed', 'replacements': manifest['replacements'],
              'changed_during_pack': manifest['changed_during_pack']}
    save_json(Path(str(archive) + '.verification.json'), result)
    cleanup = [Path(str(archive) + suffix) for suffix in ('.resume-tail.backup', '.recovery-fragment.bin')]
    if manifest.get('recovery_backup'):
        cleanup.append(Path(manifest['recovery_backup']))
    for path in cleanup:
        assert path.resolve().parent == archive.resolve().parent
        assert path.name.startswith(archive.name + '.')
        if path.exists():
            path.unlink()
    report(stage='complete', percent=100, archive_bytes=archive.stat().st_size,
           verified_entries=count, sha256=archive_hash)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    status_path = Path(str(args.archive) + '.progress.json')
    started = time.time()

    def report(**values):
        values.update(pid=os.getpid(), updated_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                      elapsed_seconds=round(time.time() - started), archive=str(args.archive))
        save_json(status_path, values)
        print(json.dumps(values, ensure_ascii=False), flush=True)

    try:
        if not args.verify_only:
            resume(args.root, args.archive, report)
        verify(args.archive, report)
    except Exception as exc:
        report(stage='failed', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
