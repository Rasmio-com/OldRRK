import { z } from "zod";

export const createPermissionSchema = z.object({
  roleId: z.string().min(1),
  entity: z.string().min(1),
  action: z.string().min(1),
  fields: z.string().optional().nullable(),
  conditions: z.string().optional().nullable(),
  inverted: z.string().optional().nullable()
});
