"use server";

import { clearSession } from "@/lib/auth/session";
import { redirect } from "next/navigation";

export async function logoutAction() {
  clearSession();
  redirect("/login");
}
