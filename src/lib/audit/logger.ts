import { adminLog } from "@/lib/db/adminLog";
import type { SessionPayload } from "@/lib/auth/session";
import type { AppAbility, Subjects } from "@/lib/acl/ability";

const SENSITIVE_FIELDS = ["code", "otp", "password", "nationalCode"]; 

const SUBJECTS: Subjects[] = [
  "User",
  "NewPremium",
  "SubPremium",
  "Permission",
  "Role",
  "UserOTP",
  "Api",
  "AdminAuditLog",
  "all"
];

function isSubject(value: string): value is Subjects {
  return SUBJECTS.includes(value as Subjects);
}

export type AuditPayload = {
  action: string;
  entity: string;
  entityId: string;
  before?: Record<string, unknown> | null;
  after?: Record<string, unknown> | null;
  requestId: string;
  ip?: string | null;
  userAgent?: string | null;
};

export function maskSensitive(
  data: Record<string, unknown> | null | undefined,
  entity: string,
  ability?: AppAbility
) {
  if (!data) return null;
  const masked: Record<string, unknown> = {};
  Object.entries(data).forEach(([key, value]) => {
    const shouldMask = SENSITIVE_FIELDS.some((field) => key.toLowerCase().includes(field.toLowerCase()));
    if (!shouldMask) {
      masked[key] = value;
      return;
    }

    if (!ability || !isSubject(entity)) {
      masked[key] = "***";
      return;
    }

    if (!ability.can("read", entity, key) && !ability.can("manage", "all")) {
      masked[key] = "***";
      return;
    }

    masked[key] = value;
  });
  return masked;
}

export async function logAdminAction(
  session: SessionPayload,
  payload: AuditPayload,
  ability?: AppAbility
) {
  await adminLog.adminAuditLog.create({
    data: {
      actorAdminUserId: session.adminUserId,
      action: payload.action,
      entity: payload.entity,
      entityId: payload.entityId,
      beforeJson: payload.before ? JSON.stringify(payload.before) : null,
      afterJson: payload.after ? JSON.stringify(payload.after) : null,
      timestamp: new Date(),
      ip: payload.ip ?? null,
      userAgent: payload.userAgent ?? null,
      requestId: payload.requestId
    }
  });
}
