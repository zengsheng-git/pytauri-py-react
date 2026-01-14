/*
 * @Author: zengsheng 12181283
 * @Date: 2026-01-13 10:59:00
 * @LastEditors: zengsheng 12181283
 * @LastEditTime: 2026-01-14 14:11:14
 * @FilePath: \pytauri-py-react\src\Test.tsx
 */
import {
  Button,
  Select,
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@pikoloo/darwin-ui";
// import "@pikoloo/darwin-ui/styles";
import MacTest from "./MacTest";
import {
  Table,
  TableHead,
  TableBody,
  TableRow,
  TableHeaderCell,
  TableCell,
} from "@pikoloo/darwin-ui";
import { Input } from "@pikoloo/darwin-ui";
import { useState } from "react";

export default function Test() {
  const [value, setValue] = useState("option1");
  return (
    <Card>
      <Select value={value} onChange={(e) => setValue(e.target.value)}>
        <option value="option1">Option 1</option>
        <option value="option2">Option 2</option>
      </Select>
      <div className="p-4 flex flex-col gap-4">
        <Input placeholder="Enter text..." />
        <Input error placeholder="Error state" />
        <Input success placeholder="Success state" />
      </div>
      <div className="text-[#fff] bg-blue-500">123456</div>
      <MacTest />
      <CardHeader>
        <CardTitle>Welcome to Darwin UI</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="component-preview flex-wrap gap-4">
          <Button variant="default">Default</Button>
          <Button variant="primary">Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="destructive">Destructive</Button>
          <Button variant="outline">Outline</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="primary" className="gap-2">
            {/* <Loader2 className="w-4 h-4 animate-spin" /> */}
            Loading
          </Button>
        </div>

        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>Name</TableHeaderCell>
              <TableHeaderCell>Status</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>John Doe</TableCell>
              <TableCell>Active</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
