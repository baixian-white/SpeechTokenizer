import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "H:\\H-CODE\\speechtokenizer";
const skillDir = "C:\\Users\\Windows11\\.codex\\plugins\\cache\\openai-primary-runtime\\presentations\\26.905.11957\\skills\\presentations";
const runtimeNode = "C:\\Users\\Windows11\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\bin\\node.exe";
const runtimeNodeModules = "C:\\Users\\Windows11\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\node\\node_modules";
const runtimeBinDir = "C:\\Users\\Windows11\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\bin\\override";
const runtimePython = "C:\\Users\\Windows11\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe";
const sourceSvg = path.join(workspaceDir, "figures", "scit_speech_figure_blueprint.svg");
const sourcePptx = path.join(workspaceDir, "figures", "ppt", "scit_speech_figure_blueprint_editable.pptx");
const buildDir = path.join(workspaceDir, ".codex-ppt-build", "scit-svg");
const stagingDir = path.join(buildDir, "enhanced-finalizer");
const sourceCopyPptx = path.join(stagingDir, "source-editable-copy.pptx");
const finalPptx = path.join(workspaceDir, "figures", "ppt", "scit_speech_figure_blueprint_editable_large.pptx");

process.env.RUNTIME_NODE = runtimeNode;
process.env.RUNTIME_NODE_MODULES = runtimeNodeModules;
process.env.RUNTIME_BIN_DIR = runtimeBinDir;
process.env.RUNTIME_PYTHON = runtimePython;
process.env.SKILL_DIR = skillDir;

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPptx), { recursive: true });
await fs.copyFile(sourcePptx, sourceCopyPptx);

const svg = await fs.readFile(sourceSvg, "utf8");

function parseAttributes(tag) {
  const attrs = {};
  const re = /([A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"([^"]*)"/gu;
  for (const match of tag.matchAll(re)) attrs[match[1]] = match[2];
  return attrs;
}

function parseCss(source) {
  const map = {};
  const styleBlock = source.match(/<style>([\s\S]*?)<\/style>/u)?.[1] ?? "";
  const ruleRe = /\.([A-Za-z0-9_-]+)\s*\{([^}]*)\}/gu;
  for (const match of styleBlock.matchAll(ruleRe)) {
    const props = {};
    for (const declaration of match[2].split(";")) {
      const index = declaration.indexOf(":");
      if (index < 0) continue;
      const key = declaration.slice(0, index).trim();
      const value = declaration.slice(index + 1).trim();
      if (key && value) props[key] = value;
    }
    map[match[1]] = props;
  }
  return map;
}

function decodeXml(text) {
  return text
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&apos;", "'")
    .replace(/&#(\d+);/gu, (_, code) => String.fromCodePoint(Number(code)))
    .replace(/&#x([0-9a-f]+);/giu, (_, code) => String.fromCodePoint(Number.parseInt(code, 16)));
}

function normalizeColor(value, fallback = "#000000") {
  if (!value || value === "none") return value ?? fallback;
  if (/^#[0-9a-f]{3}$/iu.test(value)) {
    return `#${value[1]}${value[1]}${value[2]}${value[2]}${value[3]}${value[3]}`.toUpperCase();
  }
  return value.toUpperCase();
}

const css = parseCss(svg);
function computed(attrs) {
  const classProps = attrs.class ? css[attrs.class] ?? {} : {};
  return { ...classProps, ...attrs };
}

function number(value, fallback = 0) {
  const parsed = Number.parseFloat(value ?? "");
  return Number.isFinite(parsed) ? parsed : fallback;
}

function paintFill(props) {
  const fill = props.fill ?? "none";
  if (fill === "none") return "none";
  const color = normalizeColor(fill);
  const stronger = {
    "#F7FBFF": "#EDF6FF",
    "#FBFDFF": "#F7FBFF",
    "#ECF6FF": "#E4F2FF",
    "#FFF8EA": "#FFF3DC",
    "#FFF7ED": "#FFF1E2",
    "#EFFAF4": "#E7F7EE",
    "#F4F2FF": "#ECEAFF",
  };
  return stronger[color] ?? color;
}

function paintLine(props) {
  const stroke = props.stroke ?? "none";
  if (stroke === "none") return { style: "solid", fill: "none", width: 0 };
  return {
    style: props["stroke-dasharray"] ? "dashed" : "solid",
    fill: normalizeColor(stroke),
    width: number(props["stroke-width"], 1) * 1.3,
  };
}

function parseFont(props) {
  const result = {
    family: props["font-family"]?.split(",")[0]?.trim() || "Arial",
    size: number(props["font-size"], 16),
    weight: number(props["font-weight"], 400),
    italic: props["font-style"] === "italic",
  };
  const shorthand = props.font;
  if (shorthand) {
    result.italic = /\bitalic\b/iu.test(shorthand);
    const weightMatch = shorthand.match(/\b([4-9]00)\b/u);
    if (weightMatch) result.weight = Number(weightMatch[1]);
    const sizeMatch = shorthand.match(/([0-9.]+)px\s+(.+)$/u);
    if (sizeMatch) {
      result.size = Number(sizeMatch[1]);
      result.family = sizeMatch[2].split(",")[0].trim();
    }
  }
  return result;
}

function parsePath(d, offsetX, offsetY) {
  const tokens = d.match(/[MLHVZmlhvz]|-?(?:\d+\.?\d*|\.\d+)/gu) ?? [];
  const commands = [];
  const points = [];
  let i = 0;
  let command = null;
  let currentX = 0;
  let currentY = 0;
  let startX = 0;
  let startY = 0;
  const pushPoint = (type, x, y) => {
    currentX = x;
    currentY = y;
    if (type === "M") {
      startX = x;
      startY = y;
    }
    points.push({ x: x + offsetX, y: y + offsetY });
    commands.push({ type, x: x + offsetX, y: y + offsetY });
  };
  while (i < tokens.length) {
    if (/^[MLHVZmlhvz]$/u.test(tokens[i])) command = tokens[i++];
    if (!command) throw new Error(`Invalid SVG path: ${d}`);
    const upper = command.toUpperCase();
    const relative = command !== upper;
    if (upper === "Z") {
      commands.push({ type: "Z" });
      currentX = startX;
      currentY = startY;
      command = null;
      continue;
    }
    if (upper === "M" || upper === "L") {
      const xRaw = Number(tokens[i++]);
      const yRaw = Number(tokens[i++]);
      const x = relative ? currentX + xRaw : xRaw;
      const y = relative ? currentY + yRaw : yRaw;
      pushPoint(upper, x, y);
      if (upper === "M") command = relative ? "l" : "L";
      continue;
    }
    if (upper === "H") {
      const xRaw = Number(tokens[i++]);
      const x = relative ? currentX + xRaw : xRaw;
      pushPoint("L", x, currentY);
      continue;
    }
    if (upper === "V") {
      const yRaw = Number(tokens[i++]);
      const y = relative ? currentY + yRaw : yRaw;
      pushPoint("L", currentX, y);
    }
  }
  return { commands, points };
}

const presentation = Presentation.create({ slideSize: { width: 2400, height: 1400 } });
const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";

let sequence = 0;
const counts = { rect: 0, circle: 0, path: 0, arrowhead: 0, text: 0 };
const nextName = (kind) => `svg_${kind}_${String(++sequence).padStart(3, "0")}`;
const rectRecords = [];
const circleRecords = [];

function addRect(attrs, offset) {
  const props = computed(attrs);
  let x = number(props.x) + offset.x;
  let y = number(props.y) + offset.y;
  let width = number(props.width);
  let height = number(props.height);
  if (!(width > 0 && height > 0)) return;
  const shouldExpand = ["mod", "train", "note", "lossbox"].includes(attrs.class) && width <= 420 && height <= 220;
  if (shouldExpand) {
    const scale = 1.04;
    x -= width * (scale - 1) / 2;
    y -= height * (scale - 1) / 2;
    width *= scale;
    height *= scale;
  }
  const radius = number(props.rx, 0);
  rectRecords.push({ x, y, width, height, className: attrs.class ?? "" });
  slide.shapes.add({
    geometry: radius > 0 ? "roundRect" : "rect",
    name: nextName("rect"),
    position: { left: x, top: y, width, height },
    fill: paintFill(props),
    line: paintLine(props),
    ...(radius > 0 ? { borderRadius: radius } : {}),
  });
  counts.rect += 1;
}

function addCircle(attrs, offset) {
  const props = computed(attrs);
  const r = number(props.r);
  if (!(r > 0)) return;
  const cx = number(props.cx) + offset.x;
  const cy = number(props.cy) + offset.y;
  circleRecords.push({ cx, cy, r });
  slide.shapes.add({
    geometry: "ellipse",
    name: nextName("circle"),
    position: { left: cx - r, top: cy - r, width: 2 * r, height: 2 * r },
    fill: paintFill(props),
    line: paintLine(props),
  });
  counts.circle += 1;
}

function addCustomPolyline(parsed, props, name) {
  if (parsed.points.length < 2) return;
  const xs = parsed.points.map((p) => p.x);
  const ys = parsed.points.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs);
  const maxY = Math.max(...ys);
  const width = Math.max(1, maxX - minX);
  const height = Math.max(1, maxY - minY);
  const commands = parsed.commands.map((command) => {
    if (command.type === "M") return { moveTo: { x: command.x - minX, y: command.y - minY } };
    if (command.type === "L") return { lineTo: { x: command.x - minX, y: command.y - minY } };
    return { close: {} };
  });
  slide.shapes.add({
    geometry: "custom",
    name,
    position: { left: minX, top: minY, width, height },
    fill: paintFill(props),
    line: paintLine(props),
    customPaths: [{ width, height, commands }],
  });
}

function addArrowhead(points, color) {
  if (points.length < 2) return;
  const tip = points.at(-1);
  let previousIndex = points.length - 2;
  while (previousIndex >= 0 && points[previousIndex].x === tip.x && points[previousIndex].y === tip.y) previousIndex -= 1;
  if (previousIndex < 0) return;
  const previous = points[previousIndex];
  const dx = tip.x - previous.x;
  const dy = tip.y - previous.y;
  const magnitude = Math.hypot(dx, dy);
  if (magnitude === 0) return;
  const ux = dx / magnitude;
  const uy = dy / magnitude;
  const length = 14;
  const halfWidth = 7;
  const baseX = tip.x - ux * length;
  const baseY = tip.y - uy * length;
  const px = -uy * halfWidth;
  const py = ux * halfWidth;
  const triangle = [
    tip,
    { x: baseX + px, y: baseY + py },
    { x: baseX - px, y: baseY - py },
  ];
  const xs = triangle.map((p) => p.x);
  const ys = triangle.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const width = Math.max(1, Math.max(...xs) - minX);
  const height = Math.max(1, Math.max(...ys) - minY);
  slide.shapes.add({
    geometry: "custom",
    name: nextName("arrowhead"),
    position: { left: minX, top: minY, width, height },
    fill: normalizeColor(color),
    line: { style: "solid", fill: normalizeColor(color), width: 0.5 },
    customPaths: [{
      width,
      height,
      commands: [
        { moveTo: { x: triangle[0].x - minX, y: triangle[0].y - minY } },
        { lineTo: { x: triangle[1].x - minX, y: triangle[1].y - minY } },
        { lineTo: { x: triangle[2].x - minX, y: triangle[2].y - minY } },
        { close: {} },
      ],
    }],
  });
  counts.arrowhead += 1;
}

function addPath(attrs, offset) {
  const props = computed(attrs);
  if (!props.d) return;
  const parsed = parsePath(props.d, offset.x, offset.y);
  addCustomPolyline(parsed, props, nextName("path"));
  counts.path += 1;
  if (props["marker-end"]) addArrowhead(parsed.points, props.stroke ?? "#2386d8");
}

function addText(attrs, rawText, offset) {
  const props = computed(attrs);
  let content = decodeXml(rawText).replace(/\s+/gu, " ").trim();
  if (!content) return;
  const font = parseFont(props);
  const textScale = {
    h1: 1.20,
    h2: 1.18,
    t: 1.18,
    s: 1.15,
    xs: 1.18,
    math: 1.15,
    nt: 1.08,
  }[attrs.class] ?? 1.12;
  font.size *= textScale;
  const x = number(props.x) + offset.x;
  const y = number(props.y) + offset.y;
  const anchor = props["text-anchor"] ?? "start";
  let width = Math.max(font.size * 2.2, Math.min(950, content.length * font.size * 0.68 + 18));
  let height = Math.max(24, font.size * 1.48);
  let left = x;
  let alignment = "left";
  let wrapMode = "none";
  if (anchor === "middle") {
    left = x - width / 2;
    alignment = "center";
  } else if (anchor === "end") {
    left = x - width;
    alignment = "right";
  }
  let top = props["dominant-baseline"] === "middle" ? y - height / 2 : y - font.size * 1.06;

  const containingCircle = circleRecords
    .filter((circle) => Math.hypot(x - circle.cx, y - circle.cy) <= circle.r)
    .sort((a, b) => a.r - b.r)[0];
  const containingRect = rectRecords
    .filter((rect) => x >= rect.x && x <= rect.x + rect.width && y >= rect.y && y <= rect.y + rect.height)
    .filter((rect) => rect.width * rect.height <= 100000 && rect.height <= 230)
    .sort((a, b) => a.width * a.height - b.width * b.height)[0];

  if (containingCircle) {
    left = containingCircle.cx - containingCircle.r + 3;
    top = containingCircle.cy - containingCircle.r + 3;
    width = containingCircle.r * 2 - 6;
    height = containingCircle.r * 2 - 6;
    alignment = "center";
  } else if (containingRect) {
    const paddingX = Math.min(10, Math.max(4, containingRect.width * 0.035));
    left = containingRect.x + paddingX;
    width = containingRect.width - paddingX * 2;
    height = Math.min(height, containingRect.height - 6);
    top = Math.max(containingRect.y + 3, Math.min(top, containingRect.y + containingRect.height - height - 3));
    if (anchor === "middle") alignment = "center";
  }

  if (content === "DISCRETE INDEX TRANSMISSION") {
    content = "DISCRETE INDEX\nTRANSMISSION";
    font.size = 28;
    left = 1285;
    top = 282;
    width = 405;
    height = 65;
    alignment = "left";
    wrapMode = "square";
  } else if (content === "forward data flow") {
    font.size = 15;
    left = 1120;
    top = 1340;
    width = 165;
    height = 26;
    alignment = "center";
  } else if (content === "training supervision / optimization") {
    font.size = 15;
    left = 1380;
    top = 1340;
    width = 315;
    height = 26;
    alignment = "center";
  }
  const box = slide.shapes.add({
    geometry: "textbox",
    name: nextName("text"),
    position: { left, top, width, height },
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  box.text = content;
  box.text.style = {
    typeface: font.family,
    fontSize: font.size,
    bold: font.weight >= 600,
    italic: font.italic,
    color: normalizeColor(props.fill ?? "#000000"),
    alignment,
    verticalAlignment: "middle",
    autoFit: "shrinkText",
    wrap: wrapMode,
    insets: { top: 0, right: 0, bottom: 0, left: 0 },
  };
  counts.text += 1;
}

const tokens = svg.match(/<!--[\s\S]*?-->|<style>[\s\S]*?<\/style>|<[^>]+>|[^<]+/gu) ?? [];
const offsets = [{ x: 0, y: 0 }];
let inDefs = 0;
let pendingText = null;

for (const token of tokens) {
  if (token.startsWith("<!--") || token.startsWith("<style")) continue;
  if (pendingText) {
    if (token.startsWith("</text")) {
      addText(pendingText.attrs, pendingText.content, pendingText.offset);
      pendingText = null;
    } else if (!token.startsWith("<")) {
      pendingText.content += token;
    }
    continue;
  }
  if (token.startsWith("<defs")) {
    inDefs += 1;
    continue;
  }
  if (token.startsWith("</defs")) {
    inDefs = Math.max(0, inDefs - 1);
    continue;
  }
  if (inDefs > 0) continue;
  if (token.startsWith("<g")) {
    const attrs = parseAttributes(token);
    const parent = offsets.at(-1);
    const match = attrs.transform?.match(/translate\(\s*(-?[0-9.]+)(?:[ ,]+(-?[0-9.]+))?\s*\)/u);
    offsets.push({
      x: parent.x + (match ? Number(match[1]) : 0),
      y: parent.y + (match?.[2] ? Number(match[2]) : 0),
    });
    continue;
  }
  if (token.startsWith("</g")) {
    offsets.pop();
    continue;
  }
  const offset = offsets.at(-1);
  if (token.startsWith("<rect")) addRect(parseAttributes(token), offset);
  else if (token.startsWith("<circle")) addCircle(parseAttributes(token), offset);
  else if (token.startsWith("<path")) addPath(parseAttributes(token), offset);
  else if (token.startsWith("<text")) pendingText = { attrs: parseAttributes(token), content: "", offset: { ...offset } };
}

if (pendingText) throw new Error("Unclosed SVG text element");
slide.speakerNotes.textFrame.setText(
  "Large-text revision copied from the editable PPT design. Native PowerPoint shapes were retained, typography was enlarged, module fills and outlines were strengthened, and text boxes were constrained to their parent boxes.",
);

const draftPath = path.join(stagingDir, "candidate-editable-large.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);
const preview = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(path.join(buildDir, "editable-large-draft-slide-1.png"), new Uint8Array(await preview.arrayBuffer()));

const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href,
);

const result = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath: draftPath,
  finalPath: finalPptx,
  pythonExecutable: runtimePython,
  integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu",
    "22860000,13335000",
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "scit_speech_figure_blueprint_editable_large.validation-v2.json"),
});

console.log(JSON.stringify({ sourcePptx, sourceCopyPptx, finalPptx, draftPath, counts, totalObjects: Object.values(counts).reduce((a, b) => a + b, 0), result }, null, 2));
