import { AbilityBuilder, AbilityClass, PureAbility } from "@casl/ability";
import type { Permission } from "@/generated/main";

export type Actions =
  | "manage"
  | "create"
  | "read"
  | "update"
  | "delete"
  | "assignPremium"
  | "customPremiumForm";

export type Subjects =
  | "User"
  | "NewPremium"
  | "SubPremium"
  | "Permission"
  | "Role"
  | "UserOTP"
  | "Api"
  | "AdminAuditLog"
  | "all";

export type AppAbility = PureAbility<[Actions, Subjects]>;

export function buildAbility(permissions: Permission[]): AppAbility {
  const { can, cannot, build } = new AbilityBuilder(
    PureAbility as AbilityClass<AppAbility>
  );

  permissions.forEach((permission) => {
    const conditions = permission.Conditions ?? undefined;
    const fields = permission.Fields ?? undefined;
    if (permission.Inverted) {
      cannot(permission.Action as Actions, permission.Entity as Subjects, fields as never, conditions ?? undefined);
      return;
    }

    can(permission.Action as Actions, permission.Entity as Subjects, fields as never, conditions ?? undefined);
  });

  return build({
    detectSubjectType: (subject) => subject as Subjects
  });
}
