"use client";

import { useState } from "react";
import { JalaliDatePicker } from "@/components/ui/jalali-date-picker";

export function JalaliDateField({
  name,
  defaultValue
}: {
  name: string;
  defaultValue?: string;
}) {
  const [value, setValue] = useState(defaultValue ?? "");

  return (
    <div>
      <input type="hidden" name={name} value={value} />
      <JalaliDatePicker value={value} onChange={setValue} />
    </div>
  );
}
