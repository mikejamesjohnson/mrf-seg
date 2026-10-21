import { redirect } from "next/navigation";

/** There is one page, and it is the editor. */
export default function Home() {
  redirect("/editor");
}
