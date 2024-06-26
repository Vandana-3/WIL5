import streamlit as st
import streamlit.components.v1 as components
st.set_page_config(layout='wide')
# HTML code to embed
html_code = """
<div class='tableauPlaceholder' id='viz1718895882087' style='position: relative; width: 100%; height: 100vh;'>
    <noscript>
        <a href='#'><img alt='Compliance and Risk Dashboard' src='https://public.tableau.com/static/images/Co/ComplianceandRiskDashboard/ComplianceandRiskDashboard/1_rss.png' style='border: none' /></a>
    </noscript>
    <object class='tableauViz' style='display: none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='ComplianceandRiskDashboard/ComplianceandRiskDashboard' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/Co/ComplianceandRiskDashboard/ComplianceandRiskDashboard/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
    </object>
</div>
<script type='text/javascript'>
    var divElement = document.getElementById('viz1718895882087');
    var vizElement = divElement.getElementsByTagName('object')[0];
    vizElement.style.width = '100%';
    vizElement.style.height = '100vh';
    var scriptElement = document.createElement('script');
    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
    vizElement.parentNode.insertBefore(scriptElement, vizElement);
</script>
"""

# Display the HTML in the Streamlit app
components.html(html_code, height=800)
