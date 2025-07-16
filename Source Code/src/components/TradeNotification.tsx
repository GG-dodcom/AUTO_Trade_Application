// src/components/TradeNotification.tsx
import React, { useState, useRef, useEffect } from "react";

interface TradeNotificationProps {
	action: "Buy" | "Sell" | null;
	onClose?: () => void;
}

const TradeNotification: React.FC<TradeNotificationProps> = ({
	action,
	onClose,
}) => {
	// console.log("TradeNotification rendered, action:", action); // Debug

	if (!action) return null;

	const isBuy = action === "Buy";
	const message = isBuy ? "Upward Trend" : "Downward Trend";
	const bgColor = isBuy ? "bg-buy" : "bg-sell";

	// State for position and dragging
	const [position, setPosition] = useState({
		x: window.innerWidth - 210,
		y: 60,
	}); // Initial: top-4 (16px), right-4 (max-w-xs ~ 320px)
	const [isDragging, setIsDragging] = useState(false);
	const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
	const notificationRef = useRef<HTMLDivElement>(null);

	// Window dimensions (Electron: 800x600)
	const windowWidth = window.innerWidth || 800;
	const windowHeight = window.innerHeight || 600;

	// Handle mousedown to start dragging
	const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
		if ((e.target as HTMLElement).closest("button")) {
			// console.log("Clicked close button, skipping drag"); // Debug
			return;
		}
		// console.log("Starting drag at:", e.clientX, e.clientY); // Debug
		setIsDragging(true);
		setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
		e.preventDefault(); // Prevent text selection
	};

	// Handle mousemove to update position
	const handleMouseMove = (e: MouseEvent) => {
		if (!isDragging) return;

		let newX = e.clientX - dragStart.x;
		let newY = e.clientY - dragStart.y;

		// Constrain within window bounds
		const maxX = windowWidth - (notificationRef.current?.offsetWidth || 320);
		const maxY = windowHeight - (notificationRef.current?.offsetHeight || 80);
		newX = Math.max(0, Math.min(newX, maxX));
		newY = Math.max(0, Math.min(newY, maxY));

		// console.log("Dragging to:", newX, newY); // Debug
		setPosition({ x: newX, y: newY });
	};

	// Handle mouseup to stop dragging
	const handleMouseUp = () => {
		// console.log("Stopped dragging"); // Debug
		setIsDragging(false);
	};

	// // Log notification dimensions
	// useEffect(() => {
	// 	if (notificationRef.current) {
	// 		console.log(
	// 			"Notification dimensions:",
	// 			notificationRef.current.offsetWidth,
	// 			notificationRef.current.offsetHeight
	// 		);
	// 	}
	// }, []);

	// Set up global event listeners for dragging
	useEffect(() => {
		if (isDragging) {
			window.addEventListener("mousemove", handleMouseMove);
			window.addEventListener("mouseup", handleMouseUp);
		}
		return () => {
			window.removeEventListener("mousemove", handleMouseMove);
			window.removeEventListener("mouseup", handleMouseUp);
		};
	}, [isDragging, dragStart]);

	return (
		<div
			ref={notificationRef}
			className={`fixed z-50 p-4 rounded-lg shadow-lg text-white ${bgColor} flex items-center justify-between max-w-xs animate-slide-in cursor-move`}
			style={{ top: `${position.y}px`, left: `${position.x}px` }}
			onMouseDown={handleMouseDown}
		>
			<div className="flex items-center space-x-2">
				{isBuy ? (
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="20"
						height="20"
						viewBox="0 0 24 24"
						className="flex-shrink-0 animate-bounce"
					>
						<path
							fill="currentColor"
							d="M8 15h8l-4-7"
							transform="scale(1.5) translate(-2, -2)"
						/>
					</svg>
				) : (
					<svg
						xmlns="http://www.w3.org/2000/svg"
						width="20"
						height="20"
						viewBox="0 0 24 24"
						className="flex-shrink-0 animate-bounce"
					>
						<path
							fill="currentColor"
							d="M8 9h8l-4 7"
							transform="scale(1.5) translate(-2, -2)"
						/>
					</svg>
				)}
				<span className="font-semibold">{message}</span>
			</div>
			{onClose && (
				<button
					onClick={onClose}
					className="ml-4 text-white hover:text-gray-200 focus:outline-none"
					aria-label="Close notification"
				>
					<svg
						className="w-5 h-5"
						fill="none"
						stroke="currentColor"
						viewBox="0 0 24 24"
						xmlns="http://www.w3.org/2000/svg"
					>
						<path
							strokeLinecap="round"
							strokeLinejoin="round"
							strokeWidth="2"
							d="M6 18L18 6M6 6l12 12"
						/>
					</svg>
				</button>
			)}
			<style jsx>{`
				@keyframes slide-in {
					from {
						transform: translateX(100%);
						opacity: 0;
					}
					to {
						transform: translateX(0);
						opacity: 1;
					}
				}
				.animate-slide-in {
					animation: slide-in 0.5s ease-out;
				}
				@keyframes bounce {
					0%,
					100% {
						transform: translateY(0);
					}
					50% {
						transform: translateY(-4px);
					}
				}
				.animate-bounce {
					animation: bounce 1s ease infinite;
				}
			`}</style>
		</div>
	);
};

export default TradeNotification;
