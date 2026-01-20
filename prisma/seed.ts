import bcrypt from "bcryptjs";
import { PrismaClient } from "../src/generated/main";

const prisma = new PrismaClient();

async function main() {
  const superUserRole = await prisma.role.upsert({
    where: { Name: "SuperUser" },
    update: {},
    create: { Name: "SuperUser", Description: "Full access" }
  });

  const salesManagerRole = await prisma.role.upsert({
    where: { Name: "SalesManager" },
    update: {},
    create: { Name: "SalesManager", Description: "Manage users and premium" }
  });

  const salesRole = await prisma.role.upsert({
    where: { Name: "Sales" },
    update: {},
    create: { Name: "Sales", Description: "Restricted premium management" }
  });

  await prisma.permission.createMany({
    data: [
      {
        RoleId: superUserRole.Id,
        Entity: "all",
        Action: "manage",
        Fields: null,
        Conditions: null,
        Inverted: false
      },
      {
        RoleId: salesManagerRole.Id,
        Entity: "User",
        Action: "read"
      },
      {
        RoleId: salesManagerRole.Id,
        Entity: "User",
        Action: "update",
        Fields: ["Email", "PhoneNumber", "CompanyName", "Position", "NationalCode", "Birthdate"]
      },
      {
        RoleId: salesManagerRole.Id,
        Entity: "NewPremium",
        Action: "assignPremium"
      },
      {
        RoleId: salesManagerRole.Id,
        Entity: "SubPremium",
        Action: "manage"
      },
      {
        RoleId: salesManagerRole.Id,
        Entity: "UserOTP",
        Action: "read"
      },
      {
        RoleId: salesRole.Id,
        Entity: "User",
        Action: "read",
        Fields: ["Id", "UserName", "Email", "PhoneNumber", "CompanyName"]
      },
      {
        RoleId: salesRole.Id,
        Entity: "NewPremium",
        Action: "customPremiumForm"
      },
      {
        RoleId: salesRole.Id,
        Entity: "SubPremium",
        Action: "manage"
      }
    ]
  });

  const passwordHash = await bcrypt.hash("ChangeMe123!", 10);

  const superUser = await prisma.adminUser.upsert({
    where: { UserName: "superuser" },
    update: {},
    create: {
      UserName: "superuser",
      Email: "admin@example.com",
      DisplayName: "Super User",
      PasswordHash: passwordHash,
      IsActive: true
    }
  });

  await prisma.adminUserRole.upsert({
    where: {
      AdminUserId_RoleId: {
        AdminUserId: superUser.Id,
        RoleId: superUserRole.Id
      }
    },
    update: {},
    create: {
      AdminUserId: superUser.Id,
      RoleId: superUserRole.Id
    }
  });
}

main()
  .catch((error) => {
    console.error(error);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
