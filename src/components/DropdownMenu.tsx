"use client";

import { ChevronDownIcon24Outline } from "@/assets/icons";
import { useState, useRef, useEffect } from "react";
import { usePopper } from "react-popper";

interface Props {
	content: React.ReactNode;
	leftIcon?: React.ReactNode; // Icon component
	size?: "small" | "large";
	theme?: "white" | "blue" | string; // style the button
	header: string;
	dropIcon?: boolean;
	className?: string;
	isOpen?: boolean; // Controlled open state
	setIsOpen?: (isOpen: boolean) => void; // Function to update open state
}

export const DropdownMenu: React.FC<Props> = ({
	content,
	size = "large",
	theme = "white",
	leftIcon,
	header,
	dropIcon = true,
	className,
	isOpen: controlledIsOpen,
	setIsOpen: controlledSetIsOpen,
}) => {
	// Use internal state if isOpen is not controlled
	const [internalIsOpen, setInternalIsOpen] = useState(false);

	// Determine which isOpen and setIsOpen to use
	const isOpen =
		controlledIsOpen !== undefined ? controlledIsOpen : internalIsOpen;
	const setIsOpen = controlledSetIsOpen || setInternalIsOpen;

	const dropdownRef = useRef<HTMLDivElement | null>(null);
	const buttonRef = useRef<HTMLButtonElement | null>(null);
	const popperRef = useRef<HTMLDivElement | null>(null);

	// Initialize usePopper with null references initially
	const [popperElement, setPopperElement] = useState<HTMLDivElement | null>(
		null
	);
	const { styles, attributes, update } = usePopper(
		buttonRef.current,
		popperElement,
		{
			placement: "bottom-start",
			modifiers: [
				{ name: "offset", options: { offset: [0, 8] } }, // 8px vertical offset to prevent overlap with the button
				{
					name: "flip",
					options: {
						// Prioritize horizontal flipping first, then vertical
						fallbackPlacements: ["bottom-end", "top-start", "top-end"],
						padding: 8,
					},
				},
				{
					name: "preventOverflow",
					options: {
						padding: 8, // Keep 8px padding from viewport edges
						rootBoundary: "viewport", // Use the viewport as the boundary
						mainAxis: true, // Prevent overflow on the main axis (vertical for bottom/top)
						altAxis: true, // Prevent overflow on the alternate axis (horizontal for start/end)
					},
				},
			],
		}
	);

	// Update the popper element reference when the popperRef changes
	useEffect(() => {
		if (isOpen) setPopperElement(popperRef.current);
		else setPopperElement(null);
	}, [isOpen]);

	// Force popper to update after the popper element is mounted or when isOpen changes
	useEffect(() => {
		if (isOpen && update) update();
	}, [isOpen, popperElement, update]);

	// Handle window resize to update popper position
	useEffect(() => {
		const handleResize = () => isOpen && update && update();
		window.addEventListener("resize", handleResize);
		return () => window.removeEventListener("resize", handleResize);
	}, [isOpen, update]);

	// Use ResizeObserver to update popper position when the pop-out size changes
	useEffect(() => {
		if (!isOpen || !popperRef.current || !update) return;
		const observer = new ResizeObserver(() => update());
		observer.observe(popperRef.current);
		return () => observer.disconnect();
	}, [isOpen, update]);

	// Close dropdown when clicking outside
	useEffect(() => {
		const handleClickOutside = (event: MouseEvent) => {
			if (
				dropdownRef.current &&
				!dropdownRef.current.contains(event.target as Node)
			) {
				setIsOpen(false);
			}
		};
		if (isOpen) document.addEventListener("mousedown", handleClickOutside);
		return () => document.removeEventListener("mousedown", handleClickOutside);
	}, [isOpen, setIsOpen]);

	// Debug positioning with a delay to ensure the pop-out is fully rendered
	// useEffect(() => {
	// 	if (!isOpen) return;

	// 	const timer = setTimeout(() => {
	// 		console.log("Button ref:", buttonRef.current);
	// 		console.log("Popper ref:", popperRef.current);
	// 		if (buttonRef.current && popperRef.current) {
	// 			const buttonRect = buttonRef.current.getBoundingClientRect();
	// 			const popperRect = popperRef.current.getBoundingClientRect();
	// 			console.log("Button position:", {
	// 				top: buttonRect.top,
	// 				left: buttonRect.left,
	// 				right: buttonRect.right,
	// 				bottom: buttonRect.bottom,
	// 			});
	// 			console.log("Popper position:", {
	// 				top: popperRect.top,
	// 				left: popperRect.left,
	// 				right: popperRect.right,
	// 				bottom: popperRect.bottom,
	// 				width: popperRect.width,
	// 				height: popperRect.height,
	// 			});
	// 			console.log("Viewport:", {
	// 				width: window.innerWidth,
	// 				height: window.innerHeight,
	// 			});
	// 		}
	// 	}, 100); // Delay to ensure the pop-out is fully rendered

	// 	return () => clearTimeout(timer);
	// }, [isOpen]);

	// useEffect(() => {
	// 	if (isOpen) {
	// 		console.log("Popper styles:", styles.popper);
	// 		console.log("Popper attributes:", attributes.popper);
	// 	}
	// }, [isOpen, styles, attributes]);

	return (
		<div
			className={`relative inline-block text-left flex ${className}`}
			ref={dropdownRef}
		>
			{/* Dropdown Button */}
			<button
				ref={buttonRef}
				onClick={() => setIsOpen(!isOpen)}
				className={`
                    gap-2 rounded-lg font-medium focus:outline-none 
                    text-center inline-flex justify-center items-center
                    ${size === "small" && "px-3 py-2 text-xs"}
                    ${size === "large" && "px-5 py-2.5 text-sm"}
                    ${
											theme === "blue" &&
											`text-white bg-blue-700 hover:bg-blue-800 focus:ring-4 focus:ring-blue-300 
                        dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800`
										}
                    ${
											theme === "white" &&
											"text-gray-800 border border-gray-200 hover:bg-gray-100 focus:ring-4 focus:ring-gray-300"
										}
                    ${theme}  
                `}
			>
				{/* Left Icon */}
				{leftIcon && leftIcon}
				{/* Header */}
				{header}
				{/* Dropdown Icon */}
				{dropIcon && (
					<ChevronDownIcon24Outline
						className={`
                            ${size === "small" && "w-5 h-5"}
                            ${size === "large" && "w-3 h-3"}
                        `}
					/>
				)}
			</button>

			{/* Dropdown Content */}
			{isOpen && (
				<div
					ref={popperRef}
					style={styles.popper}
					{...attributes.popper}
					className="z-10 bg-white rounded-lg shadow-sm dark:bg-gray-700 w-auto"
				>
					{content}
				</div>
			)}
		</div>
	);
};
