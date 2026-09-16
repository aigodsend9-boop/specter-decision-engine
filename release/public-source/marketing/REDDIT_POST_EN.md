# Title

I built a typed decision framework for Python agents — looking for feedback on the API and calibration boundary

# Post

I'm building **Specter Decision Engine**, a small Python framework for decision outputs that software can consume directly, instead of parsing generated prose. I used AI assistance to build and test it.

You pass one state and a list of independent typed questions:

- **Noul:** probability of yes/no.
- **Choice:** a supplied option plus its probability distribution.
- **Score:** a continuous value over an ordered rubric.

The framework snapshots the state, evaluates questions independently with bounded async concurrency, and constructs validated typed results from numeric model logits. It includes a Python SDK, CLI installation check, authenticated HTTP adapter and an OpenAPI contract.

**Important limits:** this is a framework preview, not a new trained model. You bring the numeric backend and independently validated calibration. Without compatible calibration, it returns `UNCALIBRATED`. The tests use synthetic fixtures; I am not claiming measured model accuracy, universal calibration, a custom GPU sampler or Jev-level speed. It doesn't execute actions.

The code has automated contract and integration tests, but it is not production-certified or “perfect.” The license is still pending, so please don't interpret the public repository as an open-source license or an unrestricted reuse grant.

Repository: https://github.com/gamerbooker/specter-decision-engine

After the prerelease is actually available, installation instructions and the offline `doctor` check will be in the README. The attached image is an AI-generated architecture illustration, not a benchmark or product screenshot.

I'd particularly welcome feedback on:
1. Are the three question/result types useful in your agent workflows?
2. How would you specify the calibration artifact and its domain boundary?
3. Which numeric backend should be tested first?

Disclosure: I'm the project owner. Inspired by typed decision interfaces; not affiliated with TypeSafe/Jev.

---

Publisher checklist (remove before posting): confirm the repository/prerelease URLs work; update availability wording; read the target subreddit's current self-promotion and AI-content rules; post once in an appropriate thread; answer feedback without repeated promotion. Do not claim open-source licensing while the license is pending.
