import { prisma } from "@/lib/db/prisma";

export async function getAdminUserByUsername(username: string) {
  return prisma.adminUser.findFirst({
    where: { UserName: username, IsActive: true },
    include: { Roles: true }
  });
}

export async function getAdminUserWithRoles(adminUserId: string) {
  return prisma.adminUser.findFirst({
    where: { Id: BigInt(adminUserId) },
    include: { Roles: { include: { Role: true } } }
  });
}
