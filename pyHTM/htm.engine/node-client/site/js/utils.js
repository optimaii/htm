
// Read a page's GET URL variables and return them as an object.
function getUrlQuery() {
    var query = location.search.substr(1);
    var result = {};
    query.split("&").forEach(function(part) {
        var item = part.split("=");
        result[item[0]] = decodeURIComponent(item[1]);
    });
    return result;
}

/* From http://stackoverflow.com/questions/7128675/from-green-to-red-color-depend-on-percentage */

function getGreenToRed(percent) {
    percent = 100 - percent;
    r = percent < 50 ? 255 : Math.floor(255 - (percent * 2 - 100) * 255 / 100);
    g = percent > 50 ? 255 : Math.floor((percent * 2) * 255 / 100);
    return rgbToHex(r, g, 0);
}

/* From http://stackoverflow.com/questions/5623838/rgb-to-hex-and-hex-to-rgb */

function rgbToHex(r, g, b) {
    return ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}
