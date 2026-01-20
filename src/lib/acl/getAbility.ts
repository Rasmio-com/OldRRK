import { prisma } from "@/lib/db/prisma";
import { buildAbility } from "@/lib/acl/ability";
import type { SessionPayload } from "@/lib/auth/session";

export async function getAbilityForSession(session: SessionPayload) {
  const permissions = await prisma.permission.findMany({
    where: { RoleId: { in: session.roleIds.map((id) => BigInt(id)) } }
  });

  return buildAbility(permissions);
}
