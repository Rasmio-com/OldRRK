import { PrismaClient } from "@/generated/adminlog";

const globalForAdminLog = globalThis as unknown as { adminLog?: PrismaClient };

export const adminLog =
  globalForAdminLog.adminLog ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["query", "error", "warn"] : ["error"]
  });

if (process.env.NODE_ENV !== "production") {
  globalForAdminLog.adminLog = adminLog;
}
