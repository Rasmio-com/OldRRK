import { redirect } from "next/navigation";
import type { AppAbility, Subjects, Actions } from "@/lib/acl/ability";

export function assertAbility(ability: AppAbility, action: Actions, subject: Subjects) {
  if (!ability.can(action, subject)) {
    redirect("/admin/unauthorized");
  }
}
