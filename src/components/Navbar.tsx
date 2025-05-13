"use client";

import { Bell, Moon, Logo, Bar, User } from "@/assets/icons";
import IconButton from "./IconButton";
import { DropdownMenu } from "./DropdownMenu";
import { ProfilePopup } from "./ProfilePopup";

const Navbar = () => {
	return (
		<nav className="bg-white pt-2.5 pr-5 pb-2.5 pl-5 flex items-center justify-between h-[55px]">
			<div className="flex items-center gap-2">
				<div className="relative w-8 h-8">
					<Logo />
				</div>
			</div>
			<div className="flex items-center gap-">
				<DropdownMenu
					header="2025"
					content={
						<>
							<option>2022</option>
							<option>2023</option>
						</>
					}
				/>

				{/* Dark Mode */}
				<IconButton
					Icon={Moon}
					iconClassName="w-6 h-6"
					onClick={() => alert("Moon Clicked!")}
				/>

				{/* Notification */}
				<IconButton
					Icon={Bell}
					iconClassName="w-6 h-6"
					onClick={() => alert("Bell Clicked!")}
				/>

				<div className="flex items-center gap-2">
					{/* Profile */}
					<DropdownMenu
						content={<ProfilePopup />}
						header={"0000000000: Name"}
						dropIcon={false}
						theme="hover:bg-gray-100 !focus:outline-none !focus:ring-0 !focus:ring-offset-0"
						leftIcon={
							<div>
								<User className="w-6 h-6" />
							</div>
						}
					/>
				</div>
			</div>
		</nav>
	);
};

export default Navbar;
