import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from ttkbootstrap import Style
import json

class TodoListApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Todo List App")
        self.geometry("450x450")
        style = Style(theme="flatly")
        style.configure("Custom.TEntry", foreground="grey")

        self.task_input = ttk.Entry(self, font=("tkDefaultFont", 16), width=30, style="Custom.TEntry")
        self.task_input.pack(pady=10)
        self.task_input.insert(0, "Enter your todo here...")

        self.task_input.bind("<FocusIn>", self.clear_placeholder)
        self.task_input.bind("<FocusOut>", self.restore_placeholder)

        ttk.Button(self, text="Add", command=self.add_task).pack(pady=5)

        self.task_list = tk.Listbox(self, font=("TkDefaultFont", 16), height=10, selectmode=tk.NONE)
        self.task_list.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        #all buttons
        ttk.Button(self, text="Done", style="success.TButton", command=self.mark_done).pack(side=tk.LEFT, padx=10, pady=10)
        ttk.Button(self, text="Delete", style="danger.TButton", command=self.delete_tasks).pack(side=tk.RIGHT, padx=10, pady=10)
        ttk.Button(self, text="Edit", style="warning.Tbutton", command=self.edit_tasks).pack(side=tk.LEFT, padx = 10, pady =10)
        ttk.Button(self, text="View Stats", style="info.TButton", command=self.view_stats).pack(side=tk.BOTTOM, pady=10)

        self.load_tasks()
    


    #prompts a small tab that shows the stats of completion
    def view_stats(self):
        done_count = 0
        total_count = self.task_list.size()
        for i in range(total_count):
            if self.task_list.itemcget(i, "foreground") == "green":
                done_count += 1
        messagebox.showinfo("Task Statistics", f"Total tasks: {total_count}\nCompleted tasks: {done_count}")


    #add tasks
    def add_task(self):
        task = self.task_input.get()
        if task != "Enter your todo here...":
            self.task_list.insert(tk.END, task)
            self.task_list.itemconfigure(tk.END, foreground="black")
            self.task_input.delete(0, tk.END)
            self.save_tasks()



    #marks tasks done and changes the color to green, updates the stats
    def mark_done(self):
        task_index = self.task_list.curselection()
        if task_index:
            self.task_list.itemconfigure(task_index, foreground="green")
            self.save_tasks()


    #deletes the stats from the json file
    def delete_tasks(self):
        task_index = self.task_list.curselection()
        if task_index:
            self.task_list.delete(task_index)
            self.save_tasks()


    #tab where you write 
    def clear_placeholder(self, event):
        if self.task_input.get() == "Enter your todo here...":
            self.task_input.delete(0, tk.END)
            self.task_input.configure(style="TEntry")

    def restore_placeholder(self, event):
        if self.task_input.get() == "":
            self.task_input.insert(0, "Enter your todo here...")
            self.task_input.configure(style="Custom.TEntry")



    #loads the tasks from the json file
    def load_tasks(self):
        try:
            with open("tasks.json", "r") as f:
                data = json.load(f)
                for task in data:
                    self.task_list.insert(tk.END, task["text"])
                    self.task_list.itemconfigure(tk.END, foreground=task["color"])
        except FileNotFoundError:
            pass



    #saves tasks
    def save_tasks(self):
        data = []
        for i in range(self.task_list.size()):
            text = self.task_list.get(i)
            color = self.task_list.itemcget(i, "foreground")
            data.append({"text": text, "color": color})
        with open("tasks.json", "w") as f:
            json.dump(data, f)


    #edit tasks
    def edit_tasks(self):
        task_index = self.task_list.curselection()
        if task_index:
            current_task = self.task_list.get(task_index)
            new_task = tk.simpledialog.askstring("Edit Task", "Edit your task:", initialvalue=current_task)
            if new_task is not None and new_task.strip()!= "":
                self.task_list.delete(task_index)
                self.task_list.insert(task_index, new_task)
                self.task_list.itemconfigure(task_index, foreground = "black")
                self.save_tasks()

if __name__ == '__main__':
    app = TodoListApp()
    app.mainloop()
