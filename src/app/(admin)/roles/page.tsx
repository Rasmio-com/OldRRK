import Link from "next/link";
import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { assertAbility } from "@/lib/acl/guard";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export default async function RolesPage() {
  const session = await getSession();
  if (!session) return null;
  const ability = await getAbilityForSession(session);
  assertAbility(ability, "read", "Role");

  const roles = await prisma.role.findMany({ orderBy: { Name: "asc" } });

  return (
    <Card>
      <CardHeader>
        <CardTitle>نقش‌ها</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-2">
          {roles.map((role) => (
            <li key={role.Id.toString()} className="flex items-center justify-between rounded border p-3">
              <div>
                <p className="font-medium">{role.Name}</p>
                <p className="text-sm text-slate-500">{role.Description ?? "-"}</p>
              </div>
              <Link className="text-primary hover:underline" href={`/admin/roles/${role.Id}`}>
                مدیریت دسترسی‌ها
              </Link>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}
