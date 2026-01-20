"use server";

import { headers } from "next/headers";
import { revalidatePath } from "next/cache";
import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { logAdminAction } from "@/lib/audit/logger";
import { jalaliToGregorianDate } from "@/lib/date/convert";
import {
  updateUserSchema,
  assignPremiumSchema,
  assignPremiumSalesSchema,
  createSubPremiumSchema
} from "@/lib/validation/users";

function requestMeta() {
  const headerList = headers();
  return {
    ip: headerList.get("x-forwarded-for"),
    userAgent: headerList.get("user-agent"),
    requestId: headerList.get("x-request-id") ?? crypto.randomUUID()
  };
}

export async function updateUserAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("update", "User")) throw new Error("Forbidden");

  const parsed = updateUserSchema.safeParse(Object.fromEntries(formData));
  if (!parsed.success) throw new Error("Invalid payload");

  const id = BigInt(parsed.data.id);
  const allowedFields = Object.entries(parsed.data).reduce((acc, [key, value]) => {
    if (key === "id") return acc;
    if (!ability.can("update", "User", key)) return acc;
    if (typeof value === "string" && value.length === 0) {
      acc[key] = null;
      return acc;
    }
    if (key === "Birthdate" && typeof value === "string") {
      acc[key] = jalaliToGregorianDate(value);
      return acc;
    }
    acc[key] = value;
    return acc;
  }, {} as Record<string, unknown>);

  const before = await prisma.user.findUnique({ where: { Id: id } });
  const updated = await prisma.user.update({
    where: { Id: id },
    data: allowedFields
  });

  await logAdminAction(
    session,
    {
      action: "update",
      entity: "User",
      entityId: id.toString(),
      before: before as unknown as Record<string, unknown>,
      after: updated as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/users/${id}`);
}

export async function assignPremiumAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("assignPremium", "NewPremium")) throw new Error("Forbidden");

  const parsed = assignPremiumSchema.safeParse({
    id: formData.get("id"),
    Until: formData.get("Until"),
    Package: Number(formData.get("Package")),
    NumberOfUser: Number(formData.get("NumberOfUser")),
    Description: formData.get("Description"),
    StartDate: formData.get("StartDate"),
    OrganizationTitle: formData.get("OrganizationTitle"),
    OrganizationLogo: formData.get("OrganizationLogo")
  });

  if (!parsed.success) throw new Error("Invalid payload");

  const id = BigInt(parsed.data.id);
  const data = {
    Until: jalaliToGregorianDate(parsed.data.Until),
    Package: parsed.data.Package,
    NumberOfUser: parsed.data.NumberOfUser,
    Description: parsed.data.Description,
    StartDate: jalaliToGregorianDate(parsed.data.StartDate),
    OrganizationTitle: parsed.data.OrganizationTitle,
    OrganizationLogo: parsed.data.OrganizationLogo
  };

  const before = await prisma.newPremium.findUnique({ where: { Id: id } });

  const updated = await prisma.newPremium.upsert({
    where: { Id: id },
    update: data,
    create: { Id: id, ...data }
  });

  await logAdminAction(
    session,
    {
      action: "assignPremium",
      entity: "NewPremium",
      entityId: id.toString(),
      before: before as unknown as Record<string, unknown>,
      after: updated as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/users/${id}`);
}

export async function assignPremiumSalesAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("customPremiumForm", "NewPremium")) throw new Error("Forbidden");

  const parsed = assignPremiumSalesSchema.safeParse({
    id: formData.get("id"),
    Until: formData.get("Until"),
    Package: Number(formData.get("Package")),
    NumberOfUser: Number(formData.get("NumberOfUser"))
  });

  if (!parsed.success) throw new Error("Invalid payload");

  const id = BigInt(parsed.data.id);
  const data = {
    Until: jalaliToGregorianDate(parsed.data.Until),
    Package: parsed.data.Package,
    NumberOfUser: parsed.data.NumberOfUser
  };

  const before = await prisma.newPremium.findUnique({ where: { Id: id } });

  const updated = await prisma.newPremium.upsert({
    where: { Id: id },
    update: data,
    create: { Id: id, ...data }
  });

  await logAdminAction(
    session,
    {
      action: "assignPremium",
      entity: "NewPremium",
      entityId: id.toString(),
      before: before as unknown as Record<string, unknown>,
      after: updated as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/users/${id}`);
}

export async function createSubPremiumAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("create", "SubPremium") && !ability.can("manage", "SubPremium")) {
    throw new Error("Forbidden");
  }

  const parsed = createSubPremiumSchema.safeParse({
    userId: formData.get("userId"),
    targetUserId: formData.get("targetUserId")
  });
  if (!parsed.success) throw new Error("Invalid payload");

  const record = await prisma.subPremium.create({
    data: {
      Id: BigInt(Date.now()),
      FunctorUserId: BigInt(session.adminUserId),
      TargetUserId: BigInt(parsed.data.targetUserId),
      SetDate: new Date()
    }
  });

  await logAdminAction(
    session,
    {
      action: "create",
      entity: "SubPremium",
      entityId: record.Id.toString(),
      after: record as unknown as Record<string, unknown>,
      before: null,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/users/${parsed.data.userId}`);
}

export async function deleteSubPremiumAction(formData: FormData) {
  const session = await getSession();
  if (!session) throw new Error("Unauthorized");
  const ability = await getAbilityForSession(session);
  if (!ability.can("delete", "SubPremium") && !ability.can("manage", "SubPremium")) {
    throw new Error("Forbidden");
  }

  const id = formData.get("id");
  if (!id) throw new Error("Invalid payload");

  const before = await prisma.subPremium.findUnique({ where: { Id: BigInt(id.toString()) } });
  const record = await prisma.subPremium.update({
    where: { Id: BigInt(id.toString()) },
    data: { DeleteDate: new Date() }
  });

  await logAdminAction(
    session,
    {
      action: "delete",
      entity: "SubPremium",
      entityId: record.Id.toString(),
      before: before as unknown as Record<string, unknown>,
      after: record as unknown as Record<string, unknown>,
      ...requestMeta()
    },
    ability
  );

  revalidatePath(`/admin/users/${record.TargetUserId?.toString() ?? ""}`);
}
