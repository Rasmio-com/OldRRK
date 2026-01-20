import { notFound } from "next/navigation";
import { prisma } from "@/lib/db/prisma";
import { getSession } from "@/lib/auth/session";
import { getAbilityForSession } from "@/lib/acl/getAbility";
import { assertAbility } from "@/lib/acl/guard";
import { formatJalali, formatJalaliDateTime, formatJalaliInput } from "@/lib/date/jalali";
import { updateUserAction, assignPremiumAction, assignPremiumSalesAction, createSubPremiumAction, deleteSubPremiumAction } from "@/lib/actions/users";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { JalaliDateField } from "@/components/ui/jalali-date-field";
import { adminLog } from "@/lib/db/adminLog";

export default async function UserDetailPage({ params }: { params: { id: string } }) {
  const session = await getSession();
  if (!session) return null;

  const ability = await getAbilityForSession(session);
  assertAbility(ability, "read", "User");

  const userId = BigInt(params.id);
  const user = await prisma.user.findUnique({
    where: { Id: userId },
    include: {
      NewPremium: true,
      SubPremiumAsTarget: { where: { DeleteDate: null } }
    }
  });

  if (!user) return notFound();

  const canReadOtp = ability.can("read", "UserOTP");
  const canReadOtpCode = ability.can("read", "UserOTP", "Code");

  const otp = canReadOtp
    ? await prisma.userOTP.findFirst({
        where: { UserName: user.UserName ?? "" },
        orderBy: { DateTime: "desc" }
      })
    : null;

  const canReadId = ability.can("read", "User", "Id");
  const canReadUserName = ability.can("read", "User", "UserName");
  const canReadJoinDate = ability.can("read", "User", "JoinDate");

  const editableFields = [
    { name: "Email", label: "ایمیل", type: "email" },
    { name: "PhoneNumber", label: "شماره موبایل", type: "text" },
    { name: "CompanyName", label: "نام شرکت", type: "text" },
    { name: "Position", label: "سمت", type: "text" },
    { name: "NationalCode", label: "کد ملی", type: "text" },
    { name: "Birthdate", label: "تاریخ تولد", type: "date" }
  ].filter((field) => ability.can("update", "User", field.name));

  const canAssignPremium = ability.can("assignPremium", "NewPremium");
  const canAssignPremiumSales = ability.can("customPremiumForm", "NewPremium");
  const canCreateSubPremium = ability.can("create", "SubPremium") || ability.can("manage", "SubPremium");
  const canDeleteSubPremium = ability.can("delete", "SubPremium") || ability.can("manage", "SubPremium");
  const canReadSubPremium = ability.can("read", "SubPremium") || ability.can("manage", "SubPremium");

  const canReadAuditLog = ability.can("read", "AdminAuditLog") || ability.can("manage", "all");
  const auditLogs = canReadAuditLog
    ? await adminLog.adminAuditLog.findMany({
        where: {
          Entity: "User",
          EntityId: user.Id.toString()
        },
        orderBy: { Timestamp: "desc" },
        take: 20
      })
    : [];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>اطلاعات کاربر</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {canReadId ? (
              <div>
                <Label>شناسه</Label>
                <div className="mt-2 text-sm text-slate-700">{user.Id.toString()}</div>
              </div>
            ) : null}
            {canReadUserName ? (
              <div>
                <Label>نام کاربری</Label>
                <div className="mt-2 text-sm text-slate-700">{user.UserName ?? "-"}</div>
              </div>
            ) : null}
            {canReadJoinDate ? (
              <div>
                <Label>تاریخ عضویت</Label>
                <div className="mt-2 text-sm text-slate-700">{formatJalali(user.JoinDate)}</div>
              </div>
            ) : null}
          </div>

          {editableFields.length ? (
            <form action={updateUserAction} className="mt-6 space-y-4">
              <input type="hidden" name="id" value={user.Id.toString()} />
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {editableFields.map((field) => (
                  <div key={field.name} className="space-y-2">
                    <Label htmlFor={field.name}>{field.label}</Label>
                    {field.name === "Birthdate" ? (
                      <JalaliDateField
                        name={field.name}
                        defaultValue={formatJalaliInput(user.Birthdate)}
                      />
                    ) : (
                      <Input
                        id={field.name}
                        name={field.name}
                        type={field.type}
                        defaultValue={(user as Record<string, string | null>)[field.name] ?? ""}
                      />
                    )}
                  </div>
                ))}
              </div>
              <Button type="submit">ذخیره تغییرات</Button>
            </form>
          ) : (
            <p className="mt-6 text-sm text-slate-500">شما دسترسی ویرایش ندارید.</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>اشتراک پریمیوم</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {user.NewPremium ? (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div>
                <Label>تا تاریخ</Label>
                <p className="mt-2 text-sm text-slate-700">{formatJalali(user.NewPremium.Until)}</p>
              </div>
              <div>
                <Label>پکیج</Label>
                <p className="mt-2 text-sm text-slate-700">{user.NewPremium.Package ?? "-"}</p>
              </div>
              <div>
                <Label>تعداد کاربر</Label>
                <p className="mt-2 text-sm text-slate-700">{user.NewPremium.NumberOfUser ?? "-"}</p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">اشتراکی ثبت نشده است.</p>
          )}

          {canAssignPremium ? (
            <form action={assignPremiumAction} className="space-y-4">
              <input type="hidden" name="id" value={user.Id.toString()} />
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>شروع</Label>
                  <JalaliDateField name="StartDate" defaultValue={formatJalaliInput(user.NewPremium?.StartDate ?? null)} />
                </div>
                <div className="space-y-2">
                  <Label>پایان</Label>
                  <JalaliDateField name="Until" defaultValue={formatJalaliInput(user.NewPremium?.Until ?? null)} />
                </div>
                <div className="space-y-2">
                  <Label>نوع پکیج</Label>
                  <Input name="Package" type="number" defaultValue={user.NewPremium?.Package ?? 11} />
                </div>
                <div className="space-y-2">
                  <Label>تعداد کاربر</Label>
                  <Input name="NumberOfUser" type="number" defaultValue={user.NewPremium?.NumberOfUser ?? 1} />
                </div>
                <div className="space-y-2">
                  <Label>عنوان سازمان</Label>
                  <Input name="OrganizationTitle" defaultValue={user.NewPremium?.OrganizationTitle ?? ""} />
                </div>
                <div className="space-y-2">
                  <Label>لوگوی سازمان</Label>
                  <Input name="OrganizationLogo" defaultValue={user.NewPremium?.OrganizationLogo ?? ""} />
                </div>
                <div className="space-y-2 md:col-span-2">
                  <Label>توضیحات</Label>
                  <Input name="Description" defaultValue={user.NewPremium?.Description ?? ""} />
                </div>
              </div>
              <Button type="submit">ثبت اشتراک</Button>
            </form>
          ) : null}

          {!canAssignPremium && canAssignPremiumSales ? (
            <form action={assignPremiumSalesAction} className="space-y-4">
              <input type="hidden" name="id" value={user.Id.toString()} />
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <Label>پایان</Label>
                  <JalaliDateField name="Until" defaultValue={formatJalaliInput(user.NewPremium?.Until ?? null)} />
                </div>
                <div className="space-y-2">
                  <Label>نوع پکیج</Label>
                  <Input name="Package" type="number" defaultValue={user.NewPremium?.Package ?? 11} />
                </div>
                <div className="space-y-2">
                  <Label>تعداد کاربر</Label>
                  <Input name="NumberOfUser" type="number" defaultValue={user.NewPremium?.NumberOfUser ?? 1} />
                </div>
              </div>
              <Button type="submit">ثبت اشتراک</Button>
            </form>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>زیرمجموعه‌های اشتراک</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {canCreateSubPremium ? (
            <form action={createSubPremiumAction} className="flex items-end gap-2">
              <input type="hidden" name="userId" value={user.Id.toString()} />
              <div className="space-y-2">
                <Label htmlFor="targetUserId">شناسه کاربر هدف</Label>
                <Input id="targetUserId" name="targetUserId" />
              </div>
              <Button type="submit" variant="secondary">افزودن</Button>
            </form>
          ) : (
            <p className="text-sm text-slate-500">دسترسی ثبت زیرمجموعه ندارید.</p>
          )}

          {canReadSubPremium ? (
            <div className="space-y-2">
              {user.SubPremiumAsTarget.map((item) => (
                <div key={item.Id.toString()} className="flex items-center justify-between rounded border p-3 text-sm">
                  <div>
                    <div>شناسه: {item.Id.toString()}</div>
                    <div>تاریخ ثبت: {formatJalaliDateTime(item.SetDate)}</div>
                  </div>
                  {canDeleteSubPremium ? (
                    <form action={deleteSubPremiumAction}>
                      <input type="hidden" name="id" value={item.Id.toString()} />
                      <Button type="submit" variant="destructive">حذف</Button>
                    </form>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">دسترسی مشاهده ندارید.</p>
          )}
        </CardContent>
      </Card>

      {otp ? (
        <Card>
          <CardHeader>
            <CardTitle>آخرین کد OTP</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div>
                <Label>کد</Label>
                <p className="mt-2 text-sm text-slate-700">{canReadOtpCode ? otp.Code ?? "-" : "***"}</p>
              </div>
              <div>
                <Label>زمان</Label>
                <p className="mt-2 text-sm text-slate-700">{formatJalaliDateTime(otp.DateTime)}</p>
              </div>
              <div>
                <Label>IP</Label>
                <p className="mt-2 text-sm text-slate-700">{otp.Ip ?? "-"}</p>
              </div>
              <div>
                <Label>تعداد خطا</Label>
                <p className="mt-2 text-sm text-slate-700">{otp.FailureCount ?? 0}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      ) : null}

      {canReadAuditLog ? (
        <Card>
          <CardHeader>
            <CardTitle>گزارش اقدامات مدیران</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              {auditLogs.length === 0 ? (
                <p className="text-slate-500">گزارشی برای این کاربر ثبت نشده است.</p>
              ) : (
                auditLogs.map((log) => (
                  <div key={log.Id.toString()} className="rounded border p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="font-medium">{log.Action}</span>
                      <span className="text-slate-500">{formatJalaliDateTime(log.Timestamp)}</span>
                    </div>
                    <div className="mt-1 text-slate-600">
                      <span>موجودیت: {log.Entity}</span>
                      <span className="mx-2">|</span>
                      <span>ادمین: {log.ActorAdminUserId}</span>
                    </div>
                    <div className="mt-2 grid gap-2 md:grid-cols-2">
                      <div>
                        <p className="text-xs text-slate-500">قبل</p>
                        <pre className="mt-1 whitespace-pre-wrap rounded bg-slate-50 p-2 text-xs">
                          {log.BeforeJson ?? "-"}
                        </pre>
                      </div>
                      <div>
                        <p className="text-xs text-slate-500">بعد</p>
                        <pre className="mt-1 whitespace-pre-wrap rounded bg-slate-50 p-2 text-xs">
                          {log.AfterJson ?? "-"}
                        </pre>
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
