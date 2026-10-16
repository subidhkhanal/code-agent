import * as vscode from "vscode";

/** An agent in the sidebar. Built-in agents ship with the app; custom ones are user-defined. */
export interface AgentDef {
	id: string;
	name: string;
	role: string;
	builtIn: boolean;
}

export const BUILT_IN_AGENTS: AgentDef[] = [
	{ id: "planner", name: "Planner", role: "Breaks a request into concrete steps", builtIn: true },
	{ id: "designer", name: "Designer", role: "Proposes structure and interfaces", builtIn: true },
	{ id: "coder", name: "Coder", role: "Writes and edits code", builtIn: true },
	{ id: "reviewer", name: "Reviewer", role: "Reviews changes and runs the tests", builtIn: true },
	{ id: "tester", name: "Tester", role: "Writes tests for the code", builtIn: true },
];

const CUSTOM_KEY = "codeAgent.customAgents";

class AgentsProvider implements vscode.TreeDataProvider<AgentDef> {
	private readonly changed = new vscode.EventEmitter<void>();
	readonly onDidChangeTreeData = this.changed.event;

	constructor(private readonly state: vscode.Memento) {}

	all(): AgentDef[] {
		return [...BUILT_IN_AGENTS, ...this.state.get<AgentDef[]>(CUSTOM_KEY, [])];
	}

	async add(agent: AgentDef): Promise<void> {
		const custom = this.state.get<AgentDef[]>(CUSTOM_KEY, []);
		await this.state.update(CUSTOM_KEY, [...custom, agent]);
		this.changed.fire();
	}

	getTreeItem(agent: AgentDef): vscode.TreeItem {
		const item = new vscode.TreeItem(agent.name);
		item.description = agent.role;
		item.tooltip = `${agent.name}: ${agent.role}`;
		item.iconPath = new vscode.ThemeIcon(agent.builtIn ? "hubot" : "person");
		item.contextValue = agent.builtIn ? "builtInAgent" : "customAgent";
		return item;
	}

	getChildren(): AgentDef[] {
		return this.all();
	}
}

export function activate(context: vscode.ExtensionContext): void {
	const agents = new AgentsProvider(context.globalState);
	context.subscriptions.push(
		vscode.window.registerTreeDataProvider("codeAgent.agents", agents),
		vscode.commands.registerCommand("codeAgent.newAgent", async () => {
			const name = await vscode.window.showInputBox({ prompt: "Agent name", placeHolder: "e.g. Security auditor" });
			if (!name?.trim()) return;
			const role = await vscode.window.showInputBox({
				prompt: `What should ${name.trim()} do?`,
				placeHolder: "e.g. Looks for injection and secret-handling bugs",
			});
			if (!role?.trim()) return;
			const id = `custom-${Date.now().toString(36)}`;
			await agents.add({ id, name: name.trim(), role: role.trim(), builtIn: false });
		}),
	);
}

export function deactivate(): void {}
