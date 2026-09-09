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
const sourcePng = path.join(workspaceDir, "figures", "scit_speech_figure_blueprint_4800.png");
const buildDir = path.join(workspaceDir, ".codex-ppt-build", "scit-svg");
const stagingDir = path.join(buildDir, "finalizer");
const finalPptx = path.join(workspaceDir, "figures", "ppt", "scit_speech_figure_blueprint.pptx");

process.env.RUNTIME_NODE = runtimeNode;
process.env.RUNTIME_NODE_MODULES = runtimeNodeModules;
process.env.RUNTIME_BIN_DIR = runtimeBinDir;
process.env.RUNTIME_PYTHON = runtimePython;
process.env.SKILL_DIR = skillDir;

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(stagingDir, { recursive: true });
await fs.mkdir(path.dirname(finalPptx), { recursive: true });

await fs.access(sourceSvg);
const sourceBytes = await fs.readFile(sourcePng);
const presentation = Presentation.create({
  slideSize: { width: 2400, height: 1400 },
});

const slide = presentation.slides.add();
slide.background.fill = "#FFFFFF";
slide.images.add({
  blob: sourceBytes,
  contentType: "image/png",
  alt: "SCIT-Speech system blueprint with six modules, including encoder architecture search, three-layer RVQ transmission and decoding, training, and optional token-audio speaker classification.",
  fit: "contain",
  position: { left: 0, top: 0, width: 2400, height: 1400 },
});
slide.speakerNotes.textFrame.setText(
  "Source: figures/scit_speech_figure_blueprint.svg. The full-slide image is the 4800 × 2800 high-resolution render of that SVG, used to preserve all labels, arrowheads, and layout across PowerPoint renderers.",
);

const draftPath = path.join(stagingDir, "candidate.pptx");
await (await PresentationFile.exportPptx(presentation)).save(draftPath);

const preview = await presentation.export({ slide, format: "png", scale: 1 });
await fs.writeFile(
  path.join(buildDir, "draft-slide-1.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const { finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href
);

const requirements = {
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};

const result = await finalizePresentation({
  ...requirements,
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
  requiredNativeTableOwnerSlides: [],
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, "scit_speech_figure_blueprint.validation-v3.json"),
});

console.log(JSON.stringify({ finalPptx, draftPath, result }, null, 2));
