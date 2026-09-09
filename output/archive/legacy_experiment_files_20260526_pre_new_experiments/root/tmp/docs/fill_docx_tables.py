from pathlib import Path
from shutil import copy2

from docx import Document


ROOT = Path(__file__).resolve().parents[2]


def find_docx_targets():
    paths = [
        p
        for p in ROOT.rglob("Nature_Communications_*.docx")
        if "tmp" not in p.parts and "before_table_fill" not in p.name
    ]
    source = [
        p for p in paths if not ("output" in p.parts and "doc" in p.parts)
    ][0]
    output = [
        p for p in paths if "output" in p.parts and "doc" in p.parts
    ][0]
    return source, output


def set_cell(table, row_idx, col_idx, text):
    table.cell(row_idx, col_idx).text = text


def fill_row(table, row_idx, values):
    for col_idx, text in enumerate(values):
        set_cell(table, row_idx, col_idx, text)


def main():
    source, output = find_docx_targets()
    backup_dir = ROOT / "tmp" / "docs"
    backup_dir.mkdir(parents=True, exist_ok=True)
    copy2(source, backup_dir / f"{source.stem}.pre_final_table_fill.docx")
    copy2(output, backup_dir / f"{output.stem}.pre_final_output_table_fill.docx")

    doc = Document(source)
    if len(doc.tables) < 5:
        raise RuntimeError(f"Expected at least 5 tables, found {len(doc.tables)}")

    # Table 1: payload accounting. Index rows use the 3-layer RVQ setting.
    t1 = doc.tables[0]
    table1_rows = {
        1: ["PCM 波形传输", "256000.00 bps", "2398.10 bps", "258398.10 bps", "1.00x"],
        2: ["连续潜表示传输", "1800743.75 bps", "2975.99 bps", "1803719.74 bps", "0.15x"],
        3: ["原始索引传输", "10551.23 bps", "2637.81 bps", "13189.04 bps", "20.32x"],
        4: ["16 bit 定宽承载", "2637.81 bps", "2536.35 bps", "5174.16 bps", "51.79x"],
        5: ["10 bit 打包", "1657.08 bps", "2874.53 bps", "4531.62 bps", "59.14x"],
        6: ["算术编码", "1558.90 bps", "3077.44 bps", "4636.35 bps", "57.65x"],
    }
    for row_idx, values in table1_rows.items():
        fill_row(t1, row_idx, values)

    # Table 2: payload-quality trade-off. Quality is determined by RVQ layer count.
    t2 = doc.tables[1]
    table2_rows = {
        1: ["1 层", "原始索引", "1.102", "0.714", "-14.78 dB", "6154.89 bps"],
        2: ["2 层", "原始索引", "1.398", "0.830", "-3.22 dB", "9671.96 bps"],
        3: ["3 层", "原始索引", "1.627", "0.861", "-0.63 dB", "13189.04 bps"],
        4: ["3 层", "10 bit 打包", "1.627", "0.861", "-0.63 dB", "4531.62 bps"],
        5: ["3 层", "算术编码", "1.627", "0.861", "-0.63 dB", "4636.35 bps"],
    }
    for row_idx, values in table2_rows.items():
        fill_row(t2, row_idx, values)

    # Table 3: serialization schemes. Body data uses the 3-layer RVQ setting.
    t3 = doc.tables[2]
    table3_rows = {
        1: ["原始数组字节流", "64 bit/index", "10551.23 bps", "20.00%", "低", "int64 索引数组，体积最大"],
        2: ["16 bit 定宽承载", "16 bit/index", "2637.81 bps; 78 B/chunk", "49.02%", "低", "round-trip 正确"],
        3: ["10 bit 打包", "10 bit/index", "1657.08 bps; 49 B/chunk", "63.43%", "低；pack/unpack 约 0.03 ms", "round-trip 正确，主方案"],
        4: ["算术编码", "熵相关", "1558.90 bps", "66.37%", "中到高", "理论熵下界，未实现 round-trip"],
    }
    for row_idx, values in table3_rows.items():
        fill_row(t3, row_idx, values)

    # Table 4: NAS / complexity. NAS has no retrained checkpoint yet.
    t4 = doc.tables[3]
    table4_rows = {
        1: ["基线模型", "103.68M", "17.05G / 1s", "44.05 ms/chunk", "20.52 ms/chunk", "STOI 0.861；PESQ 1.627"],
        2: ["搜索模型", "16.63M", "待重训复测", "待重训复测", "待重训复测", "无 retrained checkpoint，质量不声明"],
    }
    for row_idx, values in table4_rows.items():
        fill_row(t4, row_idx, values)

    # Table 5: realtime loopback. GPU results are excluded from formal conclusions.
    t5 = doc.tables[4]
    table5_rows = {
        1: ["CPU", "0.25 s", "1", "44.05 ms", "20.16 ms", "64.22 ms", "RTF 0.272，实时可行"],
        2: ["CPU", "0.25 s", "3", "44.05 ms", "20.52 ms", "64.57 ms", "RTF 0.273，实时可行"],
        3: ["GPU", "0.25 s", "1", "未纳入正式结论", "未纳入正式结论", "未纳入正式结论", "CUDA 不支持 RTX 5070 Ti sm_120"],
        4: ["GPU", "0.25 s", "3", "未纳入正式结论", "未纳入正式结论", "未纳入正式结论", "CUDA 不支持 RTX 5070 Ti sm_120"],
    }
    for row_idx, values in table5_rows.items():
        fill_row(t5, row_idx, values)

    doc.save(source)
    doc.save(output)

    placeholders = []
    for table_idx, table in enumerate(doc.tables, 1):
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row.cells):
                text = cell.text
                if "结果待补" in text:
                    placeholders.append((table_idx, row_idx, col_idx, text))

    if placeholders:
        for item in placeholders:
            print("UNFILLED", item)
        raise SystemExit(1)

    print(f"Updated source: {source}")
    print(f"Updated output: {output}")


if __name__ == "__main__":
    main()
