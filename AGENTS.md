# Agents in this repository

This repo bundles agents and skills that Zach DeBruine uses across all
his personal work. Anyone cloning this repo gets the full set.

## Available agents

| Agent | Path | What it does |
|------|------|--------------|
| **GrantWright** | [`agents/grantwright/AGENT.md`](agents/grantwright/AGENT.md) | Personal design & writing collaborator. Generates research posters, slide decks, and one-pagers in Zach's house style. |

## How to use these with GitHub Copilot / Claude

### GitHub Copilot Chat (VS Code)
Agent files in `agents/<name>/AGENT.md` are discoverable by Copilot
Chat. From any workspace that has this repo as a folder (or vendored
into `.github/agents/`), invoke with `@grantwright`.

### Claude Code
Symlink or copy the agent file into your `.claude/agents/` directory:

```bash
mkdir -p ~/.claude/agents
ln -s "$(pwd)/agents/grantwright/AGENT.md" ~/.claude/agents/grantwright.agent.md
```

Then invoke with `/agent grantwright`.

## Available skills

Each agent declares the skills it can call. Skills live alongside their
owning agent at `agents/<name>/skills/<skill>/SKILL.md`.

| Skill | Owner | Purpose |
|------|------|---------|
| **poster-design** | GrantWright | Generate a 48×36 in landscape research poster from a `content.yaml`. |

## Contributing a new agent or skill

1. Create `agents/<name>/AGENT.md` with YAML frontmatter (`name`, `description`, `tools`, `skills`).
2. Create `agents/<name>/skills/<skill>/SKILL.md` with the design system & procedure.
3. Add the agent + skill row to the tables above.
4. Open a PR.
