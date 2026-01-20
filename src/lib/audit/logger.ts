import { adminLog } from "@/lib/db/adminLog";
import type { SessionPayload } from "@/lib/auth/session";
import type { AppAbility } from "@/lib/acl/ability";

const SENSITIVE_FIELDS = ["code", "otp", "password", "nationalCode"]; 

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

function maskSensitive(
  data: Record<string, unknown> | null | undefined,
  ability?: AppAbility
) {
  if (!data) return null;
  const masked: Record<string, unknown> = {};
  Object.entries(data).forEach(([key, value]) => {
    const shouldMask = SENSITIVE_FIELDS.some((field) => key.toLowerCase().includes(field.toLowerCase()));
    if (shouldMask && ability && !ability.can("read", "UserOTP")) {
      masked[key] = "***";
    } else {
      masked[key] = value;
    }
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
      beforeJson: payload.before ? JSON.stringify(maskSensitive(payload.before, ability)) : null,
      afterJson: payload.after ? JSON.stringify(maskSensitive(payload.after, ability)) : null,
      timestamp: new Date(),
      ip: payload.ip ?? null,
      userAgent: payload.userAgent ?? null,
      requestId: payload.requestId
    }
  });
}
