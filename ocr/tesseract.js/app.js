const express = require("express");
const app = express();
const fs = require("fs");
const multer = require("multer");

const Tesseract = require("tesseract.js")

// Setup storage options to upload file inside upload directory
const storage = multer.diskStorage({
    destination: (req, res, cb) => {
        cb(null, "./uploads");
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage: storage }).single("avatar");

app.set("view engine", "ejs");

// Routes
app.get("/", (req, res) => {
    res.render("index");
})

app.post("/upload", (req, res) => {
    upload(req, res, err => {
        fs.readFile("./uploads/" + req.file.originalname, (err, data) => {
            if(err) return console.log("This is your error", err);
            
            // testing image:  'https://tesseract.projectnaptha.com/img/eng_bw.png'
            Tesseract.recognize(
                data,
                'eng', 
                { logger: m => console.log(m) }
              ).then(({ data: { text } }) => {
                console.log(text);
                res.send(text)
              })
        });
    });
});

// Start up server
const PORT = 5000 || process.env.PORT; 
app.listen(PORT, () => console.log("Hey I'm running on port " + PORT));