import dayjs from "dayjs";
import jalaliday from "jalaliday";

dayjs.extend(jalaliday);

export function formatJalali(date?: Date | string | null) {
  if (!date) return "-";
  return dayjs(date).calendar("jalali").locale("fa").format("YYYY/MM/DD");
}

export function formatJalaliInput(date?: Date | string | null) {
  if (!date) return "";
  return dayjs(date).calendar("jalali").locale("fa").format("YYYY/MM/DD");
}

export function formatJalaliDateTime(date?: Date | string | null) {
  if (!date) return "-";
  return dayjs(date).calendar("jalali").locale("fa").format("YYYY/MM/DD HH:mm");
}
