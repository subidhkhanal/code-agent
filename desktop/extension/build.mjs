// Bundles the extension into one file, so it ships without node_modules.
import { build } from "esbuild";

await build({
	entryPoints: ["src/extension.ts"],
	bundle: true,
	outfile: "dist/extension.js",
	platform: "node",
	target: "node22",
	format: "cjs",
	external: ["vscode"],
	sourcemap: false,
	minify: process.argv.includes("--minify"),
	logLevel: "info",
});
