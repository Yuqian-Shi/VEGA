() => {
    var items = Array.prototype.slice.call(
        document.querySelectorAll('*')
    ).map(function(element) {
        const tagName = element.tagName.toLowerCase?.() || "";
        let isClickable = (element.onclick != null);

        if (!isClickable) {
            const clickableTags = [
                "input",
                "textarea",
                "select",
                "button",
                "a",
                "iframe",
                "video",
                "object",
                "embed",
                "details"
            ];
            if (clickableTags.includes(tagName)) {
                isClickable = true;
            }
        }

        if (!isClickable) {
            const role = element.getAttribute("role");
            const clickableRoles = [
                "button",
                "tab",
                "link",
                "checkbox",
                "menuitem",
                "menuitemcheckbox",
                "menuitemradio",
                "radio",
            ];
            if (role != null && clickableRoles.includes(role.toLowerCase())) {
                isClickable = true;
            }
        }

        if (!isClickable) {
            const contentEditable = element.getAttribute("contentEditable");
            if (contentEditable != null && ["", "contenteditable", "true"].includes(contentEditable.toLowerCase())) {
                isClickable = true;
            }
        }

        if (!isClickable && element.hasAttribute("jsaction")) {
            const jsactionRules = element.getAttribute("jsaction").split(";");
            for (let jsactionRule of jsactionRules) {
                const ruleSplit = jsactionRule.trim().split(":");
                if ((ruleSplit.length >= 1) && (ruleSplit.length <= 2)) {
                    const [eventType, namespace, actionName] = ruleSplit.length === 1
                        ? ["click", ...ruleSplit[0].trim().split("."), "_"]
                        : [ruleSplit[0], ...ruleSplit[1].trim().split("."), "_"];
                    if (!isClickable) {
                        isClickable = (eventType === "click") && (namespace !== "none") && (actionName !== "_");
                    }
                }
            }
        }

        if (!isClickable) {
             if (tagName === "label")
                isClickable = (element.control != null) && !element.control.disabled;
        }

        if (!isClickable) {
             const className = element.getAttribute("class");
             if (className && className.toLowerCase().includes("button")) {
                isClickable = true;
             }
        }

        if (!isClickable) {
            const tabIndexValue = element.getAttribute("tabindex");
            const tabIndex = tabIndexValue ? parseInt(tabIndexValue) : -1;
            if (!(tabIndex < 0) && !isNaN(tabIndex)) {
                isClickable = true;
            }
        }
        
        // Check cursor as a last resort for static properties
        if (!isClickable) {
             const style = window.getComputedStyle(element);
             if (style.cursor == "pointer") {
                 isClickable = true;
             }
             if (tagName === "img" && ["zoom-in", "zoom-out"].includes(style.cursor)) {
                 isClickable = true;
             }
        }
        
        // Check usemap for img
        if (!isClickable && tagName === "img") {
            let mapName = element.getAttribute("usemap");
            if (mapName) {
                 const imgClientRects = element.getClientRects();
                 mapName = mapName.replace(/^#/, "").replace('"', '\\"');
                 const map = document.querySelector(`map[name=\"${mapName}\"]`);
                 if (map && (imgClientRects.length > 0)) isClickable = true;
            }
        }

        if (!isClickable) {
            return { include: false };
        }

        // Now calculate geometry
        var vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0);
        var vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
        
        var rects = [...element.getClientRects()].filter(bb => {
            var center_x = bb.left + bb.width / 2;
            var center_y = bb.top + bb.height / 2;
            
            if (center_x < 0 || center_x > vw || center_y < 0 || center_y > vh) return false;

            var elAtCenter = document.elementFromPoint(center_x, center_y);
        
            if (!elAtCenter) return false;
            return elAtCenter === element || element.contains(elAtCenter) 
        }).map(bb => {
            const rect = {
                left: Math.max(0, bb.left),
                top: Math.max(0, bb.top),
                right: Math.min(vw, bb.right),
                bottom: Math.min(vh, bb.bottom)
            };
            return {
                ...rect,
                width: rect.right - rect.left,
                height: rect.bottom - rect.top
            }
        });

        var area = rects.reduce((acc, rect) => acc + rect.width * rect.height, 0);
        
        if (area < 1) {
             return { include: false };
        }

        return {
            element: element,
            include: true,
            area,
            rects,
            text: element.textContent.trim().replace(/\s{2,}/g, ' ')
        };
    }).filter(item => item.include);

    items = items.filter(x => !items.some(y => x.element.contains(y.element) && !(x == y)))

    items.forEach(item => {
        item.element.classList.add('possible-clickable-element');
    });
}