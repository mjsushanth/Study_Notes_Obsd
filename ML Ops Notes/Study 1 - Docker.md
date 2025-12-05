
Some hard-won lessons from Docker & Compose. Hope they help someone else save few hours on debugging time.  
  
1. Keep `.dockerignore` strict. Exclude data, logs, checkpoints, `__pycache__`, and env folders. Every mistake here makes your image heavier.  
2. Use slim or multi-stage builds. Compile in one stage, ship a clean runtime image. Huge difference in size and startup time.  
3. Learn the YAML merge patterns like `<<: *base` and multiple inheritance `[ *base, *health ]`. They save a lot of repetition.  
4. Profiles are underrated — they let you isolate build, test, and run phases without rewriting Compose files.  
5. Build once with a dedicated profile and tag it — then let runtime services reference it via `image:` instead of rebuilding. This avoids repeated layer builds caused by anchor inheritance.  
6. Spend time with volume types — bind mounts for dev work, named volumes for persistence, anonymous for quick tests.  
7. Mount code as read-only (`:ro`) so nothing inside the container can modify it. Also helps caching.  
8. Be careful when redefining services. Adding env vars blindly can override ones inherited from the base block.  
9. Watch out for cache clutter — old layers and dangling images add up fast. List out the old images, Inspect space, Run `docker system prune -a`.  
10. Always read the logs and inspect containers directly. Most “mystery bugs” are visible if you just look inside with `docker exec`.  
11. Use terminal-based shell checks to see what’s actually inside an image. Commands like `du -sh /*` from within a running container reveal which folders are bloating your build.  
12. You can inspect layers from VS Code or the terminal using `docker image history` or `docker exec -it` into the container — this helps track which dependency or step adds unexpected weight.  
13. Even after a successful build, revisit your image. Re-run size checks, clean up temporary files, and simplify stages until the image feels minimal and predictable.