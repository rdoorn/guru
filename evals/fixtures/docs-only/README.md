# Orbit deployment runbook

Orbit is the (imaginary) order-intake service. This document describes how
a release reaches production and how it is taken back out again. There is
no code in this repository; it exists to hold the procedure.

## Environments

| Name       | Purpose                    | Deploy trigger            |
|------------|----------------------------|---------------------------|
| dev        | developer sandboxes        | every push to a branch    |
| staging    | pre-production validation  | merge to `main`           |
| production | customer traffic           | tagged release `vX.Y.Z`   |

Staging mirrors production in topology (two replicas behind the gateway,
one database) but runs on a quarter of the capacity.

## Release procedure

1. Cut the release branch `release/vX.Y.Z` from `main`.
2. Run the smoke suite against staging: `make smoke ENV=staging`.
3. Bump the version in `CHANGELOG.md` and tag: `git tag vX.Y.Z`.
4. Push the tag. The pipeline builds the image, signs it and rolls it out
   to production with a 10% canary for fifteen minutes.
5. Watch the dashboards (error rate, p99 latency, queue depth). If the
   canary stays green the rollout completes automatically.
6. Announce the release in `#orbit-releases` with a link to the changelog.

Database migrations run as a separate job *before* the canary starts and
must be backwards compatible with the previous release (expand/contract).

## Rollback

Roll back when the canary or the full rollout shows a sustained error-rate
increase above 1% or a p99 latency regression above 300 ms.

1. Stop the rollout: `orbit deploy pause production`.
2. Redeploy the previous tag: `orbit deploy production --tag vX.Y.(Z-1)`.
   This is a full (non-canary) rollout and takes about four minutes.
3. Do **not** roll back the database. Migrations are expand/contract, so
   the previous release runs against the new schema. If a migration itself
   is the cause, apply its `down` step by hand and file an incident.
4. Confirm recovery on the dashboards, then post in `#orbit-releases` that
   the release was rolled back and why.
5. Open a post-mortem issue within one working day.

Rollbacks are never performed by editing the running configuration; every
change goes through a tag so the audit trail stays intact.

## Access

Deploying and rolling back require the `orbit-deployer` role. Ask in
`#orbit-oncall` if you need it granted temporarily.

## Contacts

- On-call: rotation in PagerDuty, schedule "Orbit primary".
- Owners: the Intake team.
