// node src/app/api/seed.js

import { openDb } from "../../lib/db.js";

async function setup() {
	// Open SQLite connection
	const db = await openDb();

	// Create table
	await db.exec(`
	CREATE TABLE IF NOT EXISTS account (
	      id INT,
	    server VARCHAR(255) NOT NULL,
	    loginId VARCHAR(255) NOT NULL,
	    password VARCHAR(255) NOT NULL,
	    PRIMARY KEY (id, server, loginId),
	    FOREIGN KEY (id) REFERENCES user(id) ON DELETE CASCADE
	      );
	    `);

	await db.exec(`INSERT INTO account (id, server, loginId, password) VALUES
(1, 'server1.example.com', 'user1', 'password123'),
(2, 'server2.example.com', 'user2', 'securepass456'),
(1, 'server3.example.com', 'user3', 'mypassword789'),
(1, 'server4.example.com', 'user4', 'pass1234!'),
(2, 'server5.example.com', 'user5', 'strongPass987');
`);

	// Close connection
	await db.close();
}

setup().catch((err) => {
	console.error(err.message);
});
