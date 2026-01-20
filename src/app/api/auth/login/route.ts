import { NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { loginSchema } from "@/lib/validation/auth";
import { getAdminUserByUsername } from "@/lib/auth/adminUser";
import { createSession } from "@/lib/auth/session";

export async function POST(request: Request) {
  const body = await request.json();
  const parsed = loginSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ message: "Invalid payload" }, { status: 400 });
  }

  const user = await getAdminUserByUsername(parsed.data.username);
  if (!user) {
    return NextResponse.json({ message: "Invalid credentials" }, { status: 401 });
  }

  const valid = await bcrypt.compare(parsed.data.password, user.PasswordHash);
  if (!valid) {
    return NextResponse.json({ message: "Invalid credentials" }, { status: 401 });
  }

  const roleIds = user.Roles.map((role) => role.RoleId.toString());

  await createSession({
    adminUserId: user.Id.toString(),
    roleIds,
    email: user.Email ?? "",
    displayName: user.DisplayName ?? user.UserName
  });

  return NextResponse.json({ success: true });
}
