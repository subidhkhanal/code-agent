// Turns a VS Code OSS checkout into Code Agent before it is built:
//   node desktop/scripts/prepare.mjs <path-to-vscode-checkout>
// 1. Deep-merges desktop/product.json into the checkout's product.json (branding, installer ids,
//    the Open VSX extension gallery).
// 2. Copies the built agents extension into extensions/, so it ships as a built-in extension.
import { cpSync, existsSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const desktop = resolve(here, "..");
const vscode = resolve(process.argv[2] ?? "");
if (!existsSync(join(vscode, "product.json"))) {
	console.error(`not a VS Code checkout: ${vscode}`);
	process.exit(1);
}

function merge(base, extra) {
	for (const [key, value] of Object.entries(extra)) {
		const isObject = value && typeof value === "object" && !Array.isArray(value);
		base[key] = isObject && base[key] && typeof base[key] === "object" ? merge(base[key], value) : value;
	}
	return base;
}

const productPath = join(vscode, "product.json");
const product = JSON.parse(readFileSync(productPath, "utf8"));
const overrides = JSON.parse(readFileSync(join(desktop, "product.json"), "utf8"));
writeFileSync(productPath, JSON.stringify(merge(product, overrides), null, "\t") + "\n");
console.log(`branded ${productPath} as ${overrides.nameLong}`);

const ext = join(desktop, "extension");
if (!existsSync(join(ext, "dist", "extension.js"))) {
	console.error("build the extension first: cd desktop/extension && npm ci && npm run build");
	process.exit(1);
}
const target = join(vscode, "extensions", "code-agent");
rmSync(target, { recursive: true, force: true });
// Only the built output: VS Code's build compiles any extension folder with a tsconfig.json and
// lists npm dependencies, so sources and dev tooling stay behind.
for (const item of ["dist", "media", "README.md", "LICENSE"]) {
	if (existsSync(join(ext, item))) cpSync(join(ext, item), join(target, item), { recursive: true });
}
const pkg = JSON.parse(readFileSync(join(ext, "package.json"), "utf8"));
delete pkg.devDependencies;
delete pkg.scripts;
writeFileSync(join(target, "package.json"), JSON.stringify(pkg, null, "\t") + "\n");
console.log(`added built-in extension at ${target}`);
