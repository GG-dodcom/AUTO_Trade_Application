import React from "react";

interface Props {
	id: string;
	label: string;
	type?: string;
	placeholder?: string;
	required?: boolean;
	value: string;
	inputMode?:
		| "text"
		| "search"
		| "email"
		| "tel"
		| "url"
		| "none"
		| "numeric"
		| "decimal";
	disabled?: boolean;
	onChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}

const Input: React.FC<Props> = ({
	id,
	label,
	type = "text",
	placeholder,
	required = false,
	value,
	inputMode,
	disabled,
	onChange,
}) => {
	return (
		<div>
			<label
				htmlFor={id}
				className="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
			>
				{label}
			</label>
			<input
				type={type}
				id={id}
				className={`bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5
          dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500
        `}
				placeholder={placeholder}
				required={required}
				value={value}
				onChange={onChange}
				inputMode={inputMode}
				disabled={disabled}
			/>
		</div>
	);
};

export default Input;
