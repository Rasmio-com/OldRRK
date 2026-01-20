import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { assertAbility } from "@/lib/acl/guard";
import { createPermissionAction, deletePermissionAction } from "@/lib/actions/roles";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

export default async function RoleDetailPage({ params }: { params: { id: string } }) {
  const session = await getSession();
  if (!session) return null;

  const ability = await getAbilityForSession(session);
  assertAbility(ability, "read", "Role");

  const role = await prisma.role.findUnique({
    where: { Id: BigInt(params.id) },
    include: { Permissions: true }
  });

  if (!role) return null;

  const canManagePermissions = ability.can("manage", "Permission");

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>دسترسی‌های نقش {role.Name}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {role.Permissions.map((permission) => (
            <div key={permission.Id.toString()} className="flex items-center justify-between rounded border p-3 text-sm">
              <div>
                <div>{permission.Entity} - {permission.Action}</div>
                <div className="text-slate-500">
                  Fields: {permission.Fields ? JSON.stringify(permission.Fields) : "all"}
                </div>
              </div>
              {canManagePermissions ? (
                <form action={deletePermissionAction}>
                  <input type="hidden" name="id" value={permission.Id.toString()} />
                  <input type="hidden" name="roleId" value={role.Id.toString()} />
                  <Button type="submit" variant="destructive">حذف</Button>
                </form>
              ) : null}
            </div>
          ))}
        </CardContent>
      </Card>

      {canManagePermissions ? (
        <Card>
          <CardHeader>
            <CardTitle>افزودن دسترسی جدید</CardTitle>
          </CardHeader>
          <CardContent>
            <form action={createPermissionAction} className="space-y-4">
              <input type="hidden" name="roleId" value={role.Id.toString()} />
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="entity">موجودیت</Label>
                  <Input id="entity" name="entity" placeholder="User" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="action">اکشن</Label>
                  <Input id="action" name="action" placeholder="read" />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="fields">فیلدها (JSON)</Label>
                  <Input id="fields" name="fields" placeholder='["Email","PhoneNumber"]' />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="conditions">شرایط (JSON)</Label>
                  <Input id="conditions" name="conditions" placeholder='{"CompanyId":123}' />
                </div>
              </div>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" name="inverted" />
                دسترسی معکوس
              </label>
              <Button type="submit">ذخیره</Button>
            </form>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
