import Link from "next/link";
import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { assertAbility } from "@/lib/acl/guard";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from "@/components/ui/table";
import { formatJalali } from "@/lib/date/jalali";

const PAGE_SIZE = 20;

export default async function UsersPage({
  searchParams
}: {
  searchParams: Record<string, string | string[] | undefined>;
}) {
  const session = await getSession();
  if (!session) {
    return null;
  }

  const ability = await getAbilityForSession(session);
  assertAbility(ability, "read", "User");

  const query = typeof searchParams.q === "string" ? searchParams.q : "";
  const page = Number(searchParams.page ?? 1);
  const pageNumber = Number.isNaN(page) ? 1 : Math.max(1, page);
  const skip = (pageNumber - 1) * PAGE_SIZE;

  const where = query
    ? {
        OR: [
          { UserName: { contains: query } },
          { Email: { contains: query } },
          { PhoneNumber: { contains: query } },
          ...(Number.isNaN(Number(query))
            ? []
            : [{ Id: BigInt(query) }])
        ]
      }
    : {};

  const fieldLabels: Record<string, string> = {
    Id: "شناسه",
    UserName: "نام کاربری",
    Email: "ایمیل",
    PhoneNumber: "شماره موبایل",
    CompanyName: "شرکت",
    JoinDate: "تاریخ عضویت"
  };

  const fields = Object.keys(fieldLabels).filter((field) =>
    ability.can("read", "User", field)
  );

  const select = fields.reduce((acc, field) => {
    acc[field] = true;
    return acc;
  }, {} as Record<string, boolean>);
  select.Id = true;

  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where,
      skip,
      take: PAGE_SIZE,
      select
    }),
    prisma.user.count({ where })
  ]);

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">مدیریت کاربران</h1>
        <form className="flex items-center gap-2">
          <Input name="q" defaultValue={query} placeholder="جستجو نام، ایمیل یا تلفن" />
          <Button type="submit" variant="secondary">
            جستجو
          </Button>
        </form>
      </div>

      <div className="rounded-lg border bg-white">
        <Table>
          <TableHeader>
            <TableRow>
              {fields.map((field) => (
                <TableHead key={field}>{fieldLabels[field]}</TableHead>
              ))}
              <TableHead>جزئیات</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {users.map((user) => (
              <TableRow key={String(user.Id)}>
                {fields.map((field) => (
                  <TableCell key={field}>
                    {field === "JoinDate"
                      ? formatJalali((user as Record<string, unknown>)[field] as Date | null)
                      : String((user as Record<string, unknown>)[field] ?? "-")}
                  </TableCell>
                ))}
                <TableCell>
                  <Link
                    className="text-primary hover:underline"
                    href={`/admin/users/${user.Id}`}
                  >
                    مشاهده
                  </Link>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between text-sm text-slate-600">
        <span>
          صفحه {pageNumber} از {totalPages}
        </span>
        <div className="flex gap-2">
          <Link
            className="rounded border px-3 py-1 hover:bg-slate-100"
            href={`/admin/users?page=${Math.max(1, pageNumber - 1)}&q=${query}`}
          >
            قبلی
          </Link>
          <Link
            className="rounded border px-3 py-1 hover:bg-slate-100"
            href={`/admin/users?page=${Math.min(totalPages, pageNumber + 1)}&q=${query}`}
          >
            بعدی
          </Link>
        </div>
      </div>
    </div>
  );
}
