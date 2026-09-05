---
name: Setup problem
about: make install / make migrate / make doctor / make test did not work on your machine
title: "setup: "
labels: setup
---

Thanks for trying the project. Setup problems are almost always ours to fix, so please report them.

**First**: a `WARN` row for `private_modules` in `make doctor` is expected on a
public clone. The model and pricing modules are gitignored and never published. Everything works
without them except live projection runs (`make week-predict` and friends). See
docs/TROUBLESHOOTING.md.

**What I ran**

```
make ...
```

**What happened** (paste the output; `make doctor` never prints secrets)

```
```

**Environment**: OS, Python version (`python3 --version`), how you installed (`make install`?)
