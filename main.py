import cmd
import time
import asyncio
from colorama import Fore, Style, Back
from retrieval import qa, db_init, qa_llm
from add_documents import add_documents


class Run(cmd.Cmd):
    intro = "输入 help 或 ? 查看命令列表。"
    prompt = Fore.MAGENTA + "(BTW) "
    db = db_init()

    def default(self, line):
        """处理非命令输入，直接作为聊天消息处理"""

        if line.startswith("add"):
            cmd, arg, line = self.parseline(line)
            print(cmd, arg)
            # 检查arg是否是合法文件路径
            if arg:
                if not add_documents(self.db, arg):
                    print(Fore.RED + "Documents added failed." + Style.RESET_ALL)

        elif line != "EOF":
            asyncio.run(qa_llm(self.db, line))

        else:
            self.do_exit(line)

    def do_exit(self, arg):
        """退出程序: exit"""
        print(Fore.BLUE + "\nOver.")
        return True  # 返回True以退出cmd循环

    def do_EOF(self, line):
        """通过 Ctrl-D 退出程序"""
        print(Fore.BLUE + "\nOver.")
        return True


if __name__ == "__main__":

    Run().cmdloop()
