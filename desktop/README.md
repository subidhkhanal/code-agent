# Code Agent (desktop)

A Windows 11 desktop editor: a fork of [VS Code](https://github.com/microsoft/vscode) (MIT; see
[LICENSE-VSCODE.txt](LICENSE-VSCODE.txt)) with the agents built in.

- `product.json`: branding and installer ids, merged into VS Code's own `product.json`.
- `extension/`: the built-in agents extension (TypeScript, bundled with esbuild).
- `scripts/prepare.mjs`: applies both to a VS Code checkout before it is built.

Builds run in GitHub Actions ([`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml)):
VS Code at a pinned release is cloned, branded, compiled and packaged into a portable zip and a
per-user installer. To work on the extension, open `desktop/extension` in any VS Code and press F5.
