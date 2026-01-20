import { toGregorian } from "jalaali-js";

export function jalaliToGregorianDate(jalaliDate: string) {
  const [jy, jm, jd] = jalaliDate.split("/").map((value) => Number(value));
  if (!jy || !jm || !jd) {
    throw new Error("Invalid Jalali date");
  }
  const { gy, gm, gd } = toGregorian(jy, jm, jd);
  return new Date(`${gy}-${String(gm).padStart(2, "0")}-${String(gd).padStart(2, "0")}`);
}
