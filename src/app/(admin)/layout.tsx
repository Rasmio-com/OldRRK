import Link from "next/link";
import { getSession } from "@/lib/auth/session";
import { logoutAction } from "@/lib/actions/logout";

export default async function AdminLayout({
  children
}: {
  children: React.ReactNode;
}) {
  const session = await getSession();

  return (
    <div className="min-h-screen bg-slate-100">
      <header className="bg-white shadow">
        <div className="mx-auto flex items-center justify-between px-6 py-4">
          <div className="flex items-center gap-6">
            <span className="text-lg font-semibold">پنل مدیریت</span>
            <nav className="flex gap-4 text-sm text-slate-600">
              <Link href="/admin/users" className="hover:text-slate-900">
                کاربران
              </Link>
              <Link href="/admin/roles" className="hover:text-slate-900">
                نقش‌ها و دسترسی‌ها
              </Link>
            </nav>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span className="text-slate-600">{session?.displayName ?? ""}</span>
            <form action={logoutAction}>
              <button className="text-slate-600 hover:text-slate-900" type="submit">
                خروج
              </button>
            </form>
          </div>
        </div>
      </header>
      <main className="mx-auto max-w-6xl p-6">{children}</main>
    </div>
  );
}
