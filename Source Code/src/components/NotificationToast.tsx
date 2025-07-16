import { ExclamationCircle, Check, X } from "@/assets/icons";
import { toast, ToastContainer, ToastPosition } from "react-toastify";

// Function to Show Toast
export function showNotifyToast(
	message: string,
	type: "error" | "success",
	position: ToastPosition = "bottom-center", // Default position
	autoClose: number = 3000 // Default autoClose time
) {
	toast(message, {
		position, // Set the position of the toast
		autoClose, // Set the autoClose time
		type,
		closeOnClick: true,
		hideProgressBar: true,
		pauseOnHover: true,
		draggable: true,
		progress: undefined,
		theme: "colored",
		icon:
			type === "error" ? (
				<ExclamationCircle className="w-6 h-6" />
			) : (
				<Check className="w-6 h-6" />
			),
	});
}
