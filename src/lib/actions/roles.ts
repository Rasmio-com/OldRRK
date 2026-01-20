"use server";

import { headers } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { logAdminAction } from "@/lib/audit/logger";
import { createPermissionSchema } from "@/lib/validation/roles";

function requestMeta() {
  const headerList = headers();
  return {
    ip: headerList.get("x-forwarded-for"),
    userAgent: headerList.get("user-agent"),
    requestId: headerList.get("x-request-id") ?? crypto.randomUUID()
  };
}

export async function createPermissionAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("manage", "Permission")) throw new Error("Forbidden");

  const parsed = createPermissionSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) throw new Error("Invalid payload");

  const fields =
    parsed.data.fields && parsed.data.fields.trim().length > 0
      ? JSON.parse(parsed.data.fields)
      : null;
  const conditions =
    parsed.data.conditions && parsed.data.conditions.trim().length > 0
      ? JSON.parse(parsed.data.conditions)
      : null;
  const inverted = parsed.data.inverted === "on";

  const created = await prisma.permission.create({
    data: {
      RoleId: BigInt(parsed.data.roleId),
      Entity: parsed.data.entity,
      Action: parsed.data.action,
      Fields: fields,
      Conditions: conditions,
      Inverted: inverted
    }
  });

  await logAdminAction(
    session,
    {
      action: "create",
      entity: "Permission",
      entityId: created.Id.toString(),
      before: null,
      after: created as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/roles/${parsed.data.roleId}`);
}

export async function deletePermissionAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("manage", "Permission")) throw new Error("Forbidden");

  const id = formData.get("id");
  const roleId = formData.get("roleId");
  if (!id || !roleId) throw new Error("Invalid payload");

  const before = await prisma.permission.findUnique({ where: { Id: BigInt(id.toString()) } });
  const deleted = await prisma.permission.delete({ where: { Id: BigInt(id.toString()) } });

  await logAdminAction(
    session,
    {
      action: "delete",
      entity: "Permission",
      entityId: deleted.Id.toString(),
      before: before as unknown as Record<string, unknown>,
      after: deleted as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/roles/${roleId}`);
}
