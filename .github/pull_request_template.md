## Testing Requirements
- [ ] **Manual Testing:** I have verified these changes in a local/dev environment.
- [ ] **Reviewer Sign-off:** By checking this box, the reviewer confirms they have pulled this branch and ran the test suite locally.

**Steps to test manually:**
1. git clone git@github.com:terrastackai/terratorch.git terratorch.YOUR_BRANCH
2. cd terratorch.YOUR_BRANCH
3. git checkout terratorch.YOUR_BRANCH
4. python -m venv .venv
5. source .venv/bin/activate
6. pip install .[test]
7. pytest tests
