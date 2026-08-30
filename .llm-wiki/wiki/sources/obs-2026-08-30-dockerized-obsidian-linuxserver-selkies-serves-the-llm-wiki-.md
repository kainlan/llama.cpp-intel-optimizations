---
type: source
title: "Observation: Dockerized Obsidian (LinuxServer/Selkies) serves the LLM Wiki on LAN port 3001"
tags:
  - obsidian
  - docker
  - machine-setup
  - llm-wiki
status: observation
created: 2026-08-30
updated: 2026-08-30
slug: obs-2026-08-30-dockerized-obsidian-linuxserver-selkies-serves-the-llm-wiki-
relevance: high
observed_at: 2026-08-30T00:04:41.341Z
source_context: Setting up Dockerized Obsidian to browse the LLM Wiki vault over the LAN
---

# ⭐ Observation: Dockerized Obsidian (LinuxServer/Selkies) serves the LLM Wiki on LAN port 3001

Obsidian is available for the LLM Wiki via Docker (installed 2026-08-29). Image lscr.io/linuxserver/obsidian:latest (Selkies streaming, CPU-only, seccomp:unconfined, 2560x1440 clamped). Container name llm-wiki-obsidian; compose file at /home/kainlan/.llm-wiki-obsidian/docker-compose.yml; app config home bind-mounted at /home/kainlan/.llm-wiki-obsidian/config -> /config; the live vault is /Apps/llama.cpp/.llm-wiki/wiki -> /config/llama.cpp SYCL backend (project-specific vault name, renamed 2026-08-30 so Obsidian's title bar / vault switcher identifies this project; the path change also updated the vault entry in the container's obsidian.json). Access: https://<host>:3001 (self-signed cert; basic auth user=wiki, password in the compose file). Only the HTTPS port is exposed; LAN-only, never port-forward. The vault's .obsidian/ UI state is gitignored in the repo. To manage: cd /home/kainlan/.llm-wiki-obsidian && docker compose up -d / down / logs -f. Note: on first launch Obsidian showed the vault picker (pre-seeded obsidian.json did not auto-open in 1.13.7) — the user opened the vault once and it is remembered (open:true in obsidian.json), so it now restores automatically. The empty analyses/, requirements/, syntheses/ dirs at the vault root are wiki_bootstrap scaffolding (page-type folders not yet used), not stray files.

*Relevance: high*
*Context: Setting up Dockerized Obsidian to browse the LLM Wiki vault over the LAN*
*Tags: obsidian docker machine-setup llm-wiki*

---
*Observed: 2026-08-30T00:04:41.341Z*
