# AI Assistant Instructions

These instructions apply to automated agents, Cursor, and GitHub Copilot when working in this repository.

## For Agents
- Prefer server-side enforcement for auth/ACL (CASL abilities) and never rely on client-side checks.
- Follow the layered structure under `src/lib` (db, auth, acl, audit, validation).
- Log all admin mutations to the AdminLog database.
- Keep Jalali calendars for UI date entry and Jalali formatting for display.

## For Cursor
- Use TypeScript and Next.js App Router patterns already established.
- Keep Prisma schema changes aligned with SQL Server column types.
- Avoid adding TODOs for critical security flows; implement end-to-end.

## For GitHub Copilot
- Generate code that respects field-level permissions and CASL checks.
- Prefer server actions for mutations, validated by zod schemas.
- Ensure audit logging for create/update/delete/assignPremium actions.
