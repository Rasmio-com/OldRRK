# Admin Backoffice (Next.js + Prisma + CASL)

This repository contains a scalable admin/backoffice panel built with **Next.js App Router**, **Prisma (SQL Server)**, **CASL** authorization, **shadcn/ui + Tailwind**, and **Jalali calendar** support across the UI.

## ✅ Why these dependencies

- **Next.js / React**: App Router, server actions, and route handlers for secure server-side mutations.
- **Prisma ORM**: Models the main SQL Server database + a second AdminLog database.
- **CASL**: Entity/field/function-level authorization enforced server-side.
- **shadcn/ui + Tailwind**: Composable UI primitives for admin panels.
- **react-multi-date-picker + jalaali-js + jalaliday**: Jalali pickers and Jalali date formatting.
- **bcryptjs + jose**: Secure password hashing and HTTP-only session cookies.
- **zod**: Schema validation for server actions.

## Project structure

```
src/
  app/
    (auth)/login
    (admin)/users
    (admin)/roles
  components/ui
  lib/
    acl
    audit
    auth
    db
    validation
    date
prisma/
```

## Environment variables

Create `.env` using the following values:

```
DATABASE_URL="sqlserver://USER:PASSWORD@HOST:PORT;database=MainDb;encrypt=true;trustServerCertificate=true"
ADMINLOG_DATABASE_URL="sqlserver://USER:PASSWORD@HOST:PORT;database=AdminLog;encrypt=true;trustServerCertificate=true"
AUTH_SECRET="replace-with-strong-random-secret"
```

## Prisma schema (Main + AdminLog)

- Main database schema: `prisma/schema.prisma`
- Admin log schema: `prisma/adminlog.prisma`

Run generation/migrations:

```
# Generate Prisma clients (both databases)
npm run prisma:generate

# Run migrations (main DB)
npm run prisma:migrate

# Run migrations (AdminLog DB)
npm run prisma:migrate:adminlog
```

## Seed initial roles, permissions, and SuperUser

```
npm run seed
```

Default SuperUser credentials:
- Username: `superuser`
- Password: `ChangeMe123!`

## Run the app

```
npm run dev
```

## Key security notes

- **All authorization is enforced server-side** via CASL abilities built from DB permissions.
- **Field-level access** is enforced by checking CASL fields in list/detail forms.
- **Audit logging** writes to the AdminLog DB on every mutation.
- **Session auth** uses HTTP-only cookies and signed JWTs.

## Key files (MVP)

- Prisma schemas: `prisma/schema.prisma`, `prisma/adminlog.prisma`
- Ability builder: `src/lib/acl/ability.ts`
- Auth/session: `src/lib/auth/session.ts`
- Audit logger: `src/lib/audit/logger.ts`
- Users list & detail: `src/app/(admin)/users/page.tsx`, `src/app/(admin)/users/[id]/page.tsx`
- Permissions UI: `src/app/(admin)/roles/page.tsx`, `src/app/(admin)/roles/[id]/page.tsx`
