import { z } from "zod";

export const updateUserSchema = z.object({
  id: z.string().min(1),
  Email: z.string().email().optional().nullable(),
  PhoneNumber: z.string().optional().nullable(),
  CompanyName: z.string().optional().nullable(),
  Position: z.string().optional().nullable(),
  NationalCode: z.string().optional().nullable(),
  Birthdate: z.string().optional().nullable()
});

export const assignPremiumSchema = z.object({
  id: z.string().min(1),
  Until: z.string().min(1),
  Package: z.number().int(),
  NumberOfUser: z.number().int().min(1),
  Description: z.string().optional().nullable(),
  StartDate: z.string().min(1),
  OrganizationTitle: z.string().optional().nullable(),
  OrganizationLogo: z.string().optional().nullable()
});

export const assignPremiumSalesSchema = z.object({
  id: z.string().min(1),
  Until: z.string().min(1),
  Package: z.number().int(),
  NumberOfUser: z.number().int().min(1)
});

export const createSubPremiumSchema = z.object({
  userId: z.string().min(1),
  targetUserId: z.string().min(1)
});
