# builder/privacy.py
"""What the resource does to personal data: e-mail addresses, phone numbers and bank account numbers in a text become
placeholders, every mention of a user becomes @user (u/user on Reddit), and authors become keyed pseudonyms. Names,
postal addresses, usernames inside links and mentions run onto a word ("you@name", which reads like the obfuscated
"cr@p") are not found."""
import hashlib
import hmac
import os
import re

REDACTIONS = (  # in order: an e-mail address holds an @ that would otherwise read as a mention
    (re.compile(r"(?<![\w.+/-])(?:[\w.+-]+ _ )*[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"), "<email>"),
    (re.compile(r"(?<![\w/.-])(?:\+?\d{1,3}[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)"), "<phone>"),
    (re.compile(r"(?<![\w/.-])(?:[A-Z]{2})?\d{2,4}(?:[ -]\d{4}){5,7}(?![\w-])"), "<account>"),
    # a mention opens a token (".@name" too) and may run on into others ("@a@b"); "v@ginas" and "a@@holes" are words
    (re.compile(r"(?<![\w@/])@+(?!URL\b)\w+(?:@+(?!URL\b)\w+)*"), lambda m: re.sub(r"@+\w+", "@user", m.group())),
    (re.compile(r"<user>|@USER\b|\bMENTION\d+\b"), "@user"),  # sources' placeholders; MLMA's @URL stands for a link
    (re.compile(r"(?<![\w/])(/?u/)[\w-]{3,}"), r"\1user"),
)


def redact(text: str) -> str:
    for pattern, placeholder in REDACTIONS:
        text = pattern.sub(placeholder, text)
    return text


def pseudonym(platform: str, user: str) -> str:
    """A user's id, the same in every source and build for the same user on the same platform, and not recoverable by
    hashing guessed names, since it is keyed with the secret in the AUTHOR_KEY environment variable."""
    if not (key := os.environ.get("AUTHOR_KEY")):
        raise RuntimeError("AUTHOR_KEY is not set: author ids are keyed with it, so the build cannot continue")
    return hmac.new(key.encode(), f"{platform}:{user}".encode(), hashlib.sha256).hexdigest()[:16]
