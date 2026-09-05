---
name: Setup problem
about: make install / make migrate / make doctor / make test did not work on your machine
title: "setup: "
labels: setup
---

Thanks for trying the project. Setup problems are almost always ours to fix, so please report them.

**First**: `WARN` rows for `private_api` and `private_modules` in `make doctor` are expected on a
public clone. Those files are gitignored and never published. Everything except `make api` and
`make fullstack` works without them. See docs/TROUBLESHOOTING.md.

**What I ran**

```
make ...
```

**What happened** (paste the output; `make doctor` never prints secrets)

```
```

**Environment**: OS, Python version (`python3 --version`), how you installed (`make install`?)
