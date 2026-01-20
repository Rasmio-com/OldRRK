"use client";

import DatePicker from "react-multi-date-picker";
import persian from "react-date-object/calendars/persian";
import persian_fa from "react-date-object/locales/persian_fa";

export type JalaliDatePickerProps = {
  value?: string;
  onChange: (value: string) => void;
};

export function JalaliDatePicker({ value, onChange }: JalaliDatePickerProps) {
  return (
    <DatePicker
      calendar={persian}
      locale={persian_fa}
      value={value}
      onChange={(date) => onChange(date?.format("YYYY/MM/DD") ?? "")}
      format="YYYY/MM/DD"
      className="rmdp-shadow"
    />
  );
}
